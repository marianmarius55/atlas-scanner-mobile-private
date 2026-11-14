import ARKit
import Foundation
import simd

/// Saves LiDAR depth between photos, so a session gets LiDAR from many
/// more viewpoints than it has photos. Whenever the phone has moved
/// 10 cm or turned 10° since the last saved frame (at most 5 frames a
/// second, ARKit tracking normal), the ARKit frame's depth, confidence,
/// a small colour image and its pose are added to raw/lidar_frames/:
///
/// - frames.bin: per frame, depth (Float32, depth_width × depth_height,
///   rows packed), then confidence (UInt8, same size), then colour (RGB
///   UInt8, rgb_width × rgb_height: every 2nd depth pixel).
/// - frames.jsonl: one LiDARFrameRecord per line, with the frame's byte
///   offset in frames.bin and its pose and intrinsics as in pose.json.
///
/// No photo is taken. The export adds these frames' points to
/// points3D.bin, and the coverage view shows them. A frame is about
/// 280 KB; replaying the trajectories of three recorded sessions gave
/// 93–126 frames a minute.
final class LiDARFrameRecorder: @unchecked Sendable {
    static let folder = "raw/lidar_frames"
    static let dataFile = "frames.bin"
    static let indexFile = "frames.jsonl"
    /// Movement since the last saved frame that saves the next one.
    static let minMove: Float = 0.10
    static let minTurnDeg: Float = 10
    /// At most 5 frames a second.
    static let minInterval: TimeInterval = 0.2

    private let directory: URL
    private let coverage: CoverageMap?
    private let queue = DispatchQueue(
        label: "atlas.lidarframes", qos: .utility)
    private let lock = NSLock()

    // ARKit delegate thread only.
    private var lastPose: simd_float4x4?
    private var lastTime: TimeInterval = 0
    private var nextIndex = 0

    // Guarded by lock.
    private var stopped = false
    private var failure: String?
    private var savedFrames = 0
    private var savedBytes = 0

    // Queue only.
    private var dataHandle: FileHandle?
    private var indexHandle: FileHandle?
    private var offset: UInt64 = 0
    private let encoder = JSONEncoder()

    init(sessionDirectory: URL, coverage: CoverageMap?) {
        directory = sessionDirectory.appendingPathComponent(Self.folder)
        self.coverage = coverage
    }

    /// Called for every ARFrame on ARKit's delegate thread. Copies the
    /// frame if the phone has moved enough (well under a millisecond);
    /// the files are written in the background.
    func consider(_ frame: ARFrame) {
        guard case .normal = frame.camera.trackingState else { return }
        let pose = frame.camera.transform
        if let last = lastPose {
            guard frame.timestamp - lastTime >= Self.minInterval,
                  Self.moved(from: last, to: pose)
            else { return }
        }
        guard isRecording,
              let depth = frame.sceneDepth,
              let copy = Self.copyFrame(
                  depth: depth, image: frame.capturedImage)
        else { return }
        lastPose = pose
        lastTime = frame.timestamp

        let imageWidth = Int(frame.camera.imageResolution.width)
        let imageHeight = Int(frame.camera.imageResolution.height)
        let record = LiDARFrameRecord(
            index: nextIndex,
            timestamp: frame.timestamp,
            offset: 0,
            depthWidth: copy.depthWidth,
            depthHeight: copy.depthHeight,
            rgbWidth: copy.rgbWidth,
            rgbHeight: copy.rgbHeight,
            transformMatrix: LiDARFrameRecord.rows(pose),
            intrinsics: LiDARFrameRecord.rows(frame.camera.intrinsics),
            imageWidth: imageWidth,
            imageHeight: imageHeight)
        nextIndex += 1
        let camera = CoverageMap.DepthCamera(
            cameraToWorld: pose,
            intrinsics: frame.camera.intrinsics,
            imageWidth: imageWidth,
            imageHeight: imageHeight,
            depthWidth: copy.depthWidth,
            depthHeight: copy.depthHeight)
        queue.async { [self] in
            write(copy, record: record, camera: camera)
        }
    }

    /// Writes the queued frames and closes the files. No frame is saved
    /// after this.
    func finish() {
        lock.lock()
        stopped = true
        lock.unlock()
        queue.sync {
            try? dataHandle?.close()
            try? indexHandle?.close()
            dataHandle = nil
            indexHandle = nil
        }
    }

    /// One line for the capture log.
    func summary() -> String {
        lock.lock()
        defer { lock.unlock() }
        let mb = String(format: "%.0f", Double(savedBytes) / 1_000_000)
        var line = "LiDAR frames: \(savedFrames) (\(mb) MB)"
        if let reason = failure {
            line += ", stopped: \(reason)"
        }
        return line
    }

    // MARK: - Private

    /// A frame copied out of ARKit's buffers.
    private struct FrameCopy {
        let depthWidth: Int
        let depthHeight: Int
        /// Rows packed.
        let depth: [Float]
        let confidence: [UInt8]
        /// 0 × 0 if the camera image couldn't be read.
        let rgbWidth: Int
        let rgbHeight: Int
        let rgb: [UInt8]
    }

    private var isRecording: Bool {
        lock.lock()
        defer { lock.unlock() }
        return !stopped && failure == nil
    }

    private var hasFailed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return failure != nil
    }

    /// Not stopped by finish(): frames queued before it are still written.
    private func write(
        _ frame: FrameCopy,
        record: LiDARFrameRecord,
        camera: CoverageMap.DepthCamera
    ) {
        guard !hasFailed else { return }
        do {
            if dataHandle == nil {
                try openFiles()
            }
            guard let dataOut = dataHandle, let indexOut = indexHandle
            else { return }
            var rec = record
            rec.offset = offset
            var bytes = Data(capacity: rec.byteCount)
            frame.depth.withUnsafeBytes { bytes.append(contentsOf: $0) }
            bytes.append(contentsOf: frame.confidence)
            bytes.append(contentsOf: frame.rgb)
            try dataOut.write(contentsOf: bytes)
            offset += UInt64(bytes.count)
            // The index line goes after the data, so a line always
            // points at a complete frame.
            var line = try encoder.encode(rec)
            line.append(0x0A)
            try indexOut.write(contentsOf: line)
            lock.lock()
            savedFrames += 1
            savedBytes += bytes.count + line.count
            lock.unlock()
        } catch {
            lock.lock()
            failure = error.localizedDescription
            lock.unlock()
            return
        }
        coverage?.addLiDARFrame(
            depth: frame.depth, confidence: frame.confidence,
            camera: camera)
    }

    private func openFiles() throws {
        let fm = FileManager.default
        try fm.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let dataURL = directory.appendingPathComponent(Self.dataFile)
        let indexURL = directory.appendingPathComponent(Self.indexFile)
        for url in [dataURL, indexURL]
        where !fm.fileExists(atPath: url.path) {
            fm.createFile(atPath: url.path, contents: nil)
        }
        let data = try FileHandle(forWritingTo: dataURL)
        let index = try FileHandle(forWritingTo: indexURL)
        offset = try data.seekToEnd()
        _ = try index.seekToEnd()
        dataHandle = data
        indexHandle = index
    }

    /// True once the phone has moved minMove or turned minTurnDeg.
    private static func moved(
        from a: simd_float4x4, to b: simd_float4x4
    ) -> Bool {
        let m = PoseInterpolation.movement(from: a, to: b)
        return m.metres >= minMove || m.degrees >= minTurnDeg
    }

    private static func copyFrame(
        depth: ARDepthData, image: CVPixelBuffer
    ) -> FrameCopy? {
        let depthMap = depth.depthMap
        guard let confidenceMap = depth.confidenceMap,
              CVPixelBufferGetPixelFormatType(depthMap)
                  == kCVPixelFormatType_DepthFloat32,
              CVPixelBufferGetPixelFormatType(confidenceMap)
                  == kCVPixelFormatType_OneComponent8
        else { return nil }
        let w = CVPixelBufferGetWidth(depthMap)
        let h = CVPixelBufferGetHeight(depthMap)
        guard w > 1, h > 1,
              CVPixelBufferGetWidth(confidenceMap) == w,
              CVPixelBufferGetHeight(confidenceMap) == h,
              let values: [Float] = copyRows(
                  depthMap, rowBytes: w * 4, count: w * h),
              let confidence: [UInt8] = copyRows(
                  confidenceMap, rowBytes: w, count: w * h)
        else { return nil }
        let rgb = sampleColour(
            image, width: w / 2, height: h / 2,
            depthWidth: w, depthHeight: h)
        return FrameCopy(
            depthWidth: w, depthHeight: h,
            depth: values, confidence: confidence,
            rgbWidth: rgb.isEmpty ? 0 : w / 2,
            rgbHeight: rgb.isEmpty ? 0 : h / 2,
            rgb: rgb)
    }

    /// The buffer's pixels without row padding.
    private static func copyRows<T: Numeric>(
        _ buffer: CVPixelBuffer, rowBytes: Int, count: Int
    ) -> [T]? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let rows = CVPixelBufferGetHeight(buffer)
        guard bytesPerRow >= rowBytes,
              rows * rowBytes == count * MemoryLayout<T>.size
        else { return nil }
        var out = [T](repeating: 0, count: count)
        out.withUnsafeMutableBytes { dst in
            guard let dstBase = dst.baseAddress else { return }
            for r in 0..<rows {
                dstBase.advanced(by: r * rowBytes).copyMemory(
                    from: base.advanced(by: r * bytesPerRow),
                    byteCount: rowBytes)
            }
        }
        return out
    }

    /// RGB at every 2nd depth pixel, from ARKit's two-plane YCbCr camera
    /// image (BT.601 full range, as in Apple's ARKit Metal sample). The
    /// depth map covers the same view as the image. Empty if the image
    /// isn't two-plane.
    private static func sampleColour(
        _ image: CVPixelBuffer,
        width: Int, height: Int,
        depthWidth: Int, depthHeight: Int
    ) -> [UInt8] {
        CVPixelBufferLockBaseAddress(image, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
        guard CVPixelBufferGetPlaneCount(image) == 2,
              let yBase = CVPixelBufferGetBaseAddressOfPlane(image, 0),
              let cBase = CVPixelBufferGetBaseAddressOfPlane(image, 1)
        else { return [] }
        let yRow = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
        let cRow = CVPixelBufferGetBytesPerRowOfPlane(image, 1)
        let iw = CVPixelBufferGetWidthOfPlane(image, 0)
        let ih = CVPixelBufferGetHeightOfPlane(image, 0)
        let cw = CVPixelBufferGetWidthOfPlane(image, 1)
        let ch = CVPixelBufferGetHeightOfPlane(image, 1)
        guard iw > 0, ih > 0, cw > 0, ch > 0 else { return [] }
        let yp = yBase.assumingMemoryBound(to: UInt8.self)
        let cp = cBase.assumingMemoryBound(to: UInt8.self)

        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        var k = 0
        for j in 0..<height {
            let y = min(ih - 1, 2 * j * ih / depthHeight)
            let cy = min(ch - 1, y / 2)
            for i in 0..<width {
                let x = min(iw - 1, 2 * i * iw / depthWidth)
                let cx = min(cw - 1, x / 2)
                let luma = Float(yp[y * yRow + x])
                let cb = Float(cp[cy * cRow + 2 * cx]) - 128
                let cr = Float(cp[cy * cRow + 2 * cx + 1]) - 128
                rgb[k] = clampByte(luma + 1.402 * cr)
                rgb[k + 1] = clampByte(luma - 0.3441 * cb - 0.7141 * cr)
                rgb[k + 2] = clampByte(luma + 1.772 * cb)
                k += 3
            }
        }
        return rgb
    }

    private static func clampByte(_ v: Float) -> UInt8 {
        UInt8(max(0, min(255, v.rounded())))
    }
}

/// One line of raw/lidar_frames/frames.jsonl.
struct LiDARFrameRecord: Codable {
    let index: Int
    /// ARKit frame time, the clock of pose.json's timestamp.
    let timestamp: Double
    /// Byte offset of the frame in frames.bin.
    var offset: UInt64
    let depthWidth: Int
    let depthHeight: Int
    /// 0 when the frame has no colour.
    let rgbWidth: Int
    let rgbHeight: Int
    /// Camera-to-world, row-major, as transform_matrix in pose.json.
    let transformMatrix: [[Float]]
    /// For imageWidth × imageHeight, row-major, as in pose.json.
    let intrinsics: [[Float]]
    let imageWidth: Int
    let imageHeight: Int

    enum CodingKeys: String, CodingKey {
        case index
        case timestamp
        case offset
        case depthWidth = "depth_width"
        case depthHeight = "depth_height"
        case rgbWidth = "rgb_width"
        case rgbHeight = "rgb_height"
        case transformMatrix = "transform_matrix"
        case intrinsics
        case imageWidth = "image_width"
        case imageHeight = "image_height"
    }

    /// Size of the frame in frames.bin.
    var byteCount: Int {
        depthWidth * depthHeight * 5 + rgbWidth * rgbHeight * 3
    }

    var cameraToWorld: simd_float4x4? {
        let T = transformMatrix
        guard T.count == 4, T.allSatisfy({ $0.count == 4 }) else {
            return nil
        }
        return simd_float4x4(columns: (
            SIMD4(T[0][0], T[1][0], T[2][0], T[3][0]),
            SIMD4(T[0][1], T[1][1], T[2][1], T[3][1]),
            SIMD4(T[0][2], T[1][2], T[2][2], T[3][2]),
            SIMD4(T[0][3], T[1][3], T[2][3], T[3][3])
        ))
    }

    static func rows(_ m: simd_float4x4) -> [[Float]] {
        [
            [m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x],
            [m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y],
            [m.columns.0.z, m.columns.1.z, m.columns.2.z, m.columns.3.z],
            [m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w],
        ]
    }

    static func rows(_ m: simd_float3x3) -> [[Float]] {
        [
            [m.columns.0.x, m.columns.1.x, m.columns.2.x],
            [m.columns.0.y, m.columns.1.y, m.columns.2.y],
            [m.columns.0.z, m.columns.1.z, m.columns.2.z],
        ]
    }
}
