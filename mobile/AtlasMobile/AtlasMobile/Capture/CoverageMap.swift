import Foundation
import CoreVideo
import simd

/// LiDAR coverage of what the export gets, for the coverage view of the
/// capture screen (ARCameraView draws it as dots).
///
/// Depth maps are turned into world points with the maths of
/// COLMAPExporter.unprojectDepth and binned into 5 cm cells. They come
/// from the saved scans whose rgb.jpg passes the export's sharpness
/// check (COLMAPExporter.blurThreshold) and from the LiDAR frames that
/// LiDARFrameRecorder saves between photos. A cell is "sharp" once such
/// a scan has seen it, and "LiDAR only" while only LiDAR frames have.
/// Blurry scans add no cells: the export drops them with their depth.
/// Parts of the room without cells have no LiDAR in the export.
///
/// Depth maps are processed one at a time on a private serial queue.
/// The snapshot and stats are guarded by a lock, so SceneKit's render
/// thread can read them.
final class CoverageMap: @unchecked Sendable {
    /// Cell edge in metres.
    static let cellSize: Float = 0.05
    /// Depth range of COLMAPExporter.unprojectDepth.
    static let minDepth: Float = 0.1
    static let maxDepth: Float = 5.0
    /// Every 2nd depth pixel, so neighbouring samples are less than a
    /// cell apart out to about 5 m. (The export's points use every 4th.)
    static let depthStep = 2
    /// ARConfidenceLevel.medium. Low-confidence depth lies mostly on
    /// object edges and would scatter cells in mid-air.
    static let minConfidence: UInt8 = 1
    /// About 1000 m² of surface, about 30 MB.
    static let maxCells = 500_000
    /// Dot colours, close to SwiftUI's .green and .yellow (the legend).
    static let sharpColor = SIMD3<Float>(0.20, 0.78, 0.35)
    static let lidarOnlyColor = SIMD3<Float>(1.00, 0.80, 0.00)

    /// All cells as a point cloud, ready for SceneKit.
    struct Snapshot {
        /// Changes whenever the cells change.
        let version: Int
        let count: Int
        /// count × 3 Float32: cell centres in ARKit world coordinates.
        let positions: Data
        /// count × 3 Float32 RGB.
        let colors: Data
        /// count × UInt32 (0..<count) for a point element.
        let indices: Data

        static func empty(version: Int) -> Snapshot {
            Snapshot(
                version: version, count: 0,
                positions: Data(), colors: Data(), indices: Data())
        }
    }

    struct Stats {
        var scans = 0
        var blurryScans = 0
        var lidarFrames = 0
        var sharpCells = 0
        var lidarOnlyCells = 0
        var full = false
    }

    /// Where a depth map was taken from.
    struct DepthCamera {
        let cameraToWorld: simd_float4x4
        /// For imageWidth × imageHeight, as ARKit gives them.
        let intrinsics: simd_float3x3
        let imageWidth: Int
        let imageHeight: Int
        let depthWidth: Int
        let depthHeight: Int
    }

    private let queue = DispatchQueue(
        label: "atlas.coverage", qos: .utility)
    private let lock = NSLock()

    // Guarded by lock.
    private var generation = 0
    private var nextVersion = 1
    private var latest = Snapshot.empty(version: 0)
    private var published = Stats()

    // Used on queue only.
    private var cellIndex: [Int64: Int32] = [:]
    private var cellSharp: [Bool] = []
    private var positions: [Float] = []
    private var colors: [Float] = []
    private var indices: [UInt32] = []
    private var stats = Stats()
    private var snapshotDue = false
    private var lastSnapshotTime: TimeInterval = 0
    private var flushScheduled = false

    private static let minSnapshotInterval: TimeInterval = 0.5

    /// Empties the map for a new session. Depth maps still queued from
    /// the previous session are skipped.
    func reset() {
        lock.lock()
        generation += 1
        latest = .empty(version: nextVersion)
        nextVersion += 1
        published = Stats()
        lock.unlock()
        queue.async { [self] in
            cellIndex = [:]
            cellSharp = []
            positions = []
            colors = []
            indices = []
            stats = Stats()
            snapshotDue = false
            lastSnapshotTime = 0
        }
    }

    /// Adds a scan that DataRecorder.saveScan has just written to
    /// `directory` (raw/iphone/scan_NNN) and returns at once. The pose
    /// and intrinsics are taken from the frame, as saveScan saved them
    /// (pose.json is rewritten later by the 360 updates); rgb.jpg,
    /// depth.bin and confidence.bin are read back in the background.
    func addScan(directory: URL, frame: ARKitFrameData) {
        let camera = DepthCamera(
            cameraToWorld: frame.pose,
            intrinsics: frame.intrinsics,
            imageWidth: Int(frame.imageResolution.width),
            imageHeight: Int(frame.imageResolution.height),
            depthWidth: CVPixelBufferGetWidth(frame.depthMap),
            depthHeight: CVPixelBufferGetHeight(frame.depthMap))
        let gen = currentGeneration()
        queue.async { [self] in
            guard isCurrent(gen) else { return }
            processScan(directory: directory, camera: camera, generation: gen)
        }
    }

    /// Adds a LiDAR frame (no photo) and returns at once. `depth` and
    /// `confidence` are depthWidth × depthHeight with packed rows.
    func addLiDARFrame(
        depth: [Float], confidence: [UInt8], camera: DepthCamera
    ) {
        let gen = currentGeneration()
        queue.async { [self] in
            guard isCurrent(gen) else { return }
            let samples = DepthSamples(
                values: depth, row: camera.depthWidth,
                confidence: confidence, confidenceRow: camera.depthWidth)
            let changed = insertAll(
                Self.worldPoints(samples, camera: camera), sharp: false)
            stats.lidarFrames += 1
            publish(changed: changed, generation: gen)
        }
    }

    /// The latest point cloud. Cheap; safe on any thread.
    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    /// One line for the capture log.
    func summary() -> String {
        lock.lock()
        let s = published
        lock.unlock()
        return "coverage: \(s.scans) scans (\(s.blurryScans) blurry),"
            + " \(s.lidarFrames) LiDAR frames;"
            + " \(s.sharpCells) cells with a sharp photo,"
            + " \(s.lidarOnlyCells) LiDAR only"
            + (s.full ? ", map full" : "")
    }

    // MARK: - Private

    /// A depth map with rows `row` values apart, and its confidence.
    private struct DepthSamples {
        let values: [Float]
        let row: Int
        /// Empty when there is no confidence map.
        let confidence: [UInt8]
        let confidenceRow: Int
    }

    /// Cell keys hold 21 bits per axis: ±2^20 cells, ±52 km.
    private static let keyOffset: Int64 = 1 << 20

    private func currentGeneration() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return generation
    }

    private func isCurrent(_ gen: Int) -> Bool {
        currentGeneration() == gen
    }

    private func processScan(
        directory: URL, camera: DepthCamera, generation gen: Int
    ) {
        // As in the export, a photo that can't be scored is kept.
        let score = COLMAPExporter.blurScore(
            imageURL: directory.appendingPathComponent("rgb.jpg"))
        let sharp = score.map {
            $0 >= COLMAPExporter.blurThreshold
        } ?? true
        stats.scans += 1
        var changed = false
        if !sharp {
            // The export drops a blurry scan with its depth, so the
            // LiDAR of that view only counts through the LiDAR frames.
            stats.blurryScans += 1
        } else if let samples = Self.loadDepth(
            directory: directory, camera: camera)
        {
            changed = insertAll(
                Self.worldPoints(samples, camera: camera), sharp: true)
        }
        publish(changed: changed, generation: gen)
    }

    /// Publishes the stats, and a new snapshot if the cells changed. At
    /// most two snapshots a second (LiDAR frames come up to five a
    /// second, and each snapshot rebuilds the SceneKit point cloud); a
    /// change that has to wait is published by a delayed call.
    private func publish(changed: Bool, generation gen: Int) {
        if changed {
            snapshotDue = true
        }
        // Copy the arrays before taking the lock, so the render thread
        // never waits for the copy.
        var data: (count: Int, positions: Data, colors: Data, indices: Data)?
        let now = ProcessInfo.processInfo.systemUptime
        if snapshotDue {
            if now - lastSnapshotTime >= Self.minSnapshotInterval {
                data = snapshotData()
                snapshotDue = false
                lastSnapshotTime = now
            } else if !flushScheduled {
                flushScheduled = true
                queue.asyncAfter(
                    deadline: .now() + Self.minSnapshotInterval
                ) { [self] in
                    flushScheduled = false
                    guard isCurrent(gen) else { return }
                    publish(changed: false, generation: gen)
                }
            }
        }
        lock.lock()
        defer { lock.unlock() }
        guard gen == generation else { return }
        if let data {
            latest = Snapshot(
                version: nextVersion, count: data.count,
                positions: data.positions, colors: data.colors,
                indices: data.indices)
            nextVersion += 1
        }
        published = stats
    }

    private func snapshotData() -> (
        count: Int, positions: Data, colors: Data, indices: Data
    ) {
        (
            cellSharp.count,
            positions.withUnsafeBufferPointer { Data(buffer: $0) },
            colors.withUnsafeBufferPointer { Data(buffer: $0) },
            indices.withUnsafeBufferPointer { Data(buffer: $0) }
        )
    }

    /// Returns true if the map changed.
    private func insertAll(_ points: [SIMD3<Float>], sharp: Bool) -> Bool {
        var changed = false
        for p in points {
            if insert(p, sharp: sharp) {
                changed = true
            }
        }
        return changed
    }

    /// Bins one world point. Returns true if the map changed.
    private func insert(_ p: SIMD3<Float>, sharp: Bool) -> Bool {
        let c = (p / Self.cellSize).rounded(.down)
        // Also rejects NaN.
        let limit = Float(Self.keyOffset)
        guard abs(c.x) < limit, abs(c.y) < limit, abs(c.z) < limit
        else { return false }
        let key = ((Int64(c.x) + Self.keyOffset) << 42)
            | ((Int64(c.y) + Self.keyOffset) << 21)
            | (Int64(c.z) + Self.keyOffset)

        if let i = cellIndex[key] {
            let idx = Int(i)
            guard sharp, !cellSharp[idx] else { return false }
            cellSharp[idx] = true
            colors[3 * idx] = Self.sharpColor.x
            colors[3 * idx + 1] = Self.sharpColor.y
            colors[3 * idx + 2] = Self.sharpColor.z
            stats.lidarOnlyCells -= 1
            stats.sharpCells += 1
            return true
        }

        guard cellSharp.count < Self.maxCells else {
            stats.full = true
            return false
        }
        let idx = cellSharp.count
        cellIndex[key] = Int32(idx)
        indices.append(UInt32(idx))
        cellSharp.append(sharp)
        let centre = (c + 0.5) * Self.cellSize
        positions.append(centre.x)
        positions.append(centre.y)
        positions.append(centre.z)
        let color = sharp ? Self.sharpColor : Self.lidarOnlyColor
        colors.append(color.x)
        colors.append(color.y)
        colors.append(color.z)
        if sharp {
            stats.sharpCells += 1
        } else {
            stats.lidarOnlyCells += 1
        }
        return true
    }

    /// A saved scan's depth.bin and confidence.bin. Nil if the depth
    /// can't be read.
    private static func loadDepth(
        directory: URL, camera: DepthCamera
    ) -> DepthSamples? {
        let h = camera.depthHeight
        guard h > 0,
              let depthData = try? Data(
                  contentsOf: directory.appendingPathComponent("depth.bin"))
        else { return nil }
        // saveScan writes height × bytesPerRow, so rows may be padded.
        let depth: [Float] = depthData.withUnsafeBytes {
            Array($0.bindMemory(to: Float.self))
        }
        var confidence: [UInt8] = []
        if let data = try? Data(
            contentsOf: directory.appendingPathComponent("confidence.bin"))
        {
            confidence = [UInt8](data)
        }
        return DepthSamples(
            values: depth, row: depth.count / h,
            confidence: confidence, confidenceRow: confidence.count / h)
    }

    /// World points of a depth map, computed as in
    /// COLMAPExporter.unprojectDepth (before its COLMAP axis change).
    private static func worldPoints(
        _ d: DepthSamples, camera: DepthCamera
    ) -> [SIMD3<Float>] {
        let w = camera.depthWidth
        let h = camera.depthHeight
        guard w > 0, h > 0, camera.imageWidth > 0, camera.imageHeight > 0,
              d.row >= w, d.values.count >= (h - 1) * d.row + w
        else { return [] }
        let useConfidence = d.confidenceRow >= w
            && d.confidence.count >= (h - 1) * d.confidenceRow + w

        // The intrinsics are for the RGB image; scale them to the depth map.
        let K = camera.intrinsics
        let sx = Float(w) / Float(camera.imageWidth)
        let sy = Float(h) / Float(camera.imageHeight)
        let fx = K.columns.0.x * sx
        let fy = K.columns.1.y * sy
        let cx = K.columns.2.x * sx
        let cy = K.columns.2.y * sy
        guard fx > 0, fy > 0 else { return [] }

        let T = camera.cameraToWorld
        var points: [SIMD3<Float>] = []
        points.reserveCapacity((w / depthStep + 1) * (h / depthStep + 1))
        for row in stride(from: 0, to: h, by: depthStep) {
            for col in stride(from: 0, to: w, by: depthStep) {
                let z = d.values[row * d.row + col]
                guard z > minDepth, z < maxDepth else { continue }
                if useConfidence,
                   d.confidence[row * d.confidenceRow + col] < minConfidence
                {
                    continue
                }
                let p = T * SIMD4<Float>(
                    (Float(col) - cx) / fx * z,
                    -(Float(row) - cy) / fy * z,
                    -z, 1)
                points.append(SIMD3(p.x, p.y, p.z))
            }
        }
        return points
    }
}
