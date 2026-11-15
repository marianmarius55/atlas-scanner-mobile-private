import Foundation
import simd

/// What the capture overlay tells the user about the next 360 shot
/// (PanoramaGuide.hint(for:phone:)).
struct PanoramaHint: Equatable {
    let instruction: String
    let detail: String
    /// Clockwise degrees from where the phone points to the target, shown
    /// as an arrow while walking to it.
    let arrowDegrees: Double?
}

/// Guides the 360 shots of a room: where to stand, and the three heights
/// to shoot at there.
///
/// Gaussian splatting wants each place shot at levels: the 360 camera up
/// high, at chest height and near the floor. A spot is a place to stand
/// with those three shots. The room comes from the LiDAR map (CoverageMap's
/// 5 cm cells, from the saved scans and the LiDAR frames), so spots are
/// planned live for whatever room is being scanned:
/// - the floor is the lowest strong height band of the map; a place to
///   stand is floor the LiDAR has seen with nothing above it, joined to
///   where the phone has been;
/// - a 360 shot sees a room cell when it's within 4 m and nothing nearer is
///   in the way (an equirect depth map of the map, as the photo would be);
/// - a cell counts as covered once 2 spots see it; the next spot is the
///   free place that would add most, nearer places first. A spot that's
///   started is finished first.
///
/// Shots are grouped into spots by position and into levels by height
/// above the floor. A shot is good when its photo arrived and the phone was
/// still enough for the export to keep it.
///
/// Planning runs on a private queue about once a second while the guide is
/// shown. The guide only displays: it doesn't take, time or change any 360
/// shot. Tested in Python (pano_levels_sim.py) on sessions 2026-11-14 20:55
/// and 21:58: 6-7 spots saw 85-90% of the reachable room from 2+ spots.
final class PanoramaGuide: @unchecked Sendable {
    // MARK: - Tuning

    /// The map's 5 cm cells are merged into planning cells of this size.
    static let planCell: Float = 0.10
    /// 5 cm cells a planning cell needs; one alone is often a stray point.
    static let minMembers = 2
    /// A 360 shot counts as seeing a cell up to this far.
    static let maxView: Float = 4.0
    /// Gain falls from full at this distance to 0.3 at maxView.
    static let fullWeightDistance: Float = 2.5
    /// Equirect depth map of a shot position: 1.4° per pixel.
    static let mapWidth = 256
    static let mapHeight = 128
    /// Spacing of the places tried as the next spot (wider in big rooms,
    /// to try at most about 60).
    static let lattice: Float = 0.3
    static let maxCandidates: Float = 60
    /// No LiDAR cell this close to a shot position: the camera is in the
    /// open, not inside furniture or a wall.
    static let clearRadius: Float = 0.25
    /// LiDAR cells this close to the floor height are floor.
    static let floorBand: Float = 0.08
    /// The floor's LiDAR noise reaches about this high: below it nothing
    /// blocks standing.
    static let floorNoiseTop: Float = 0.35
    /// Cells below this height don't block the low shot.
    static let lowShotIgnore: Float = 0.4
    /// Where one stands, nothing is between floorNoiseTop and this height.
    static let standTop: Float = 1.6
    /// Shots this close (horizontally) to a spot's centre belong to it.
    static let spotRadius: Float = 0.4
    /// A new spot this far from spots that have a good shot.
    static let spotSpacing: Float = 0.6
    /// An unfinished spot this close to the phone is finished first.
    static let finishRadius: Float = 1.0
    /// The phone this close (horizontally) to the target is there.
    static let arriveRadius: Float = 0.3
    /// Heights above the floor of the high, chest and low shot until the
    /// user's own good shots give them (session 21:58: 1.99, 0.96, 0.19 m).
    static let defaultLevelHeights: [Float] = [1.95, 1.0, 0.25]
    static let levelNames = ["high", "chest", "low"]
    /// Dot colours: seen from 2+ spots, from 1, from none yet, and cells
    /// no free place can see.
    static let seenTwiceColor = SIMD3<Float>(0.20, 0.78, 0.35)
    static let seenOnceColor = SIMD3<Float>(1.00, 0.80, 0.00)
    static let unseenColor = SIMD3<Float>(1.00, 0.25, 0.20)
    static let unreachableColor = SIMD3<Float>(0.45, 0.45, 0.45)

    /// The level of a shot `h` metres above the floor: 0 high, 1 chest,
    /// 2 low.
    static func level(forHeight h: Float) -> Int {
        h > 1.55 ? 0 : (h < 0.65 ? 2 : 1)
    }

    // MARK: - Types

    enum Status { case needMap, planning, covered }

    enum LevelState { case todo, done, failed, blocked }

    struct Shot {
        /// Where the 360 camera was (the phone, ARKit world).
        let position: SIMD3<Float>
        /// The photo arrived and the phone turned slowly enough for the
        /// export to keep it (COLMAPExporter.x5MaxMeanDps).
        let good: Bool
    }

    struct SpotMarker: Equatable {
        /// x, z of where to hold the camera.
        let center: SIMD2<Float>
        /// World height of the high, chest and low shot.
        let levelY: [Float]
        let states: [LevelState]

        var isComplete: Bool {
            !states.contains(where: { $0 == .todo || $0 == .failed })
        }

        /// The next level to shoot: high, then chest, then low.
        var nextLevel: Int? {
            states.firstIndex(where: { $0 == .todo || $0 == .failed })
        }
    }

    struct Snapshot {
        let version: Int
        /// Changes only when the dots change.
        let dotsVersion: Int
        let status: Status
        let floorY: Float?
        /// Spots with shots.
        let spots: [SpotMarker]
        /// Where to go now: an unfinished spot, or the next new spot.
        let target: SpotMarker?
        let targetIsNew: Bool
        /// A 360 shot in flight, where it was sent from.
        let pending: SIMD3<Float>?
        let shotCount: Int
        let movedCount: Int
        /// Share of the cells free places can see that 2+ spots see.
        let coveredFraction: Float
        /// The planning cells as dots: dotCount x 3 Float32 positions and
        /// colours, and dotCount UInt32 indices.
        let dotCount: Int
        let positions: Data
        let colors: Data
        let indices: Data

        static func empty(version: Int) -> Snapshot {
            Snapshot(
                version: version, dotsVersion: version, status: .needMap,
                floorY: nil, spots: [], target: nil, targetIsNew: false,
                pending: nil, shotCount: 0, movedCount: 0,
                coveredFraction: 0, dotCount: 0,
                positions: Data(), colors: Data(), indices: Data())
        }
    }

    // MARK: - State

    private let queue = DispatchQueue(
        label: "atlas.panoramaguide", qos: .utility)
    private let lock = NSLock()

    // Guarded by lock.
    private weak var coverage: CoverageMap?
    private weak var trajectory: TrajectoryRecorder?
    private var shots: [Shot] = []
    private var pending: (position: SIMD3<Float>, since: TimeInterval)?
    private var shotsVersion = 0
    private var latest = Snapshot.empty(version: 0)
    private var nextVersion = 1
    private var generation = 0
    private var timer: DispatchSourceTimer?

    /// A place tried as the next spot.
    private struct Candidate {
        let center: SIMD2<Float>
        let levelY: [Float]
        let blocked: [Bool]
        var gain: Float
    }

    /// A plan from one state of the map and the shots.
    private struct Plan {
        let id: Int
        let status: Status
        let floorY: Float?
        let spots: [SpotMarker]
        let candidates: [Candidate]
        let shotCount: Int
        let movedCount: Int
        let coveredFraction: Float
        let dotCount: Int
        let positions: Data
        let colors: Data
        let indices: Data
    }

    /// A cell in range of a shot position, for the depth test.
    private struct Projected {
        let index: Int32
        let u: Float
        let v: Float
        let r: Float
    }

    // Queue only.
    private var plan: Plan?
    private var planCounter = 0
    private var plannedCoverageVersion = -1
    private var plannedCellCount = 0
    private var plannedShotsVersion = -1
    private var plannedAt: TimeInterval = 0
    private var previousTarget: SIMD2<Float>?
    private var publishedPlanID = -1
    private var publishedTarget: SpotMarker?
    private var publishedTargetIsNew = false
    private var publishedPending: SIMD3<Float>?
    private var depthMap = [Float](
        repeating: .infinity,
        count: PanoramaGuide.mapWidth * PanoramaGuide.mapHeight)
    private var projected: [Projected] = []
    /// Distance to each planning cell from the last shot position
    /// visibleDistances looked from, .infinity where it doesn't see it.
    private var visible: [Float] = []

    // MARK: - Use

    /// Starts a new session: no shots, and the map and trajectory to read.
    func reset(coverage: CoverageMap, trajectory: TrajectoryRecorder?) {
        lock.lock()
        self.coverage = coverage
        self.trajectory = trajectory
        shots = []
        pending = nil
        shotsVersion += 1
        generation += 1
        latest = .empty(version: nextVersion)
        nextVersion += 1
        lock.unlock()
        queue.async { [self] in
            plan = nil
            plannedCoverageVersion = -1
            plannedCellCount = 0
            plannedShotsVersion = -1
            plannedAt = 0
            previousTarget = nil
            publishedPlanID = -1
            publishedTarget = nil
            publishedTargetIsNew = false
            publishedPending = nil
        }
    }

    /// Plans about once a second while on (the guide is shown).
    func setActive(_ on: Bool) {
        lock.lock()
        defer { lock.unlock() }
        if on {
            guard timer == nil else { return }
            let t = DispatchSource.makeTimerSource(queue: queue)
            t.schedule(
                deadline: .now(), repeating: 1.0,
                leeway: .milliseconds(200))
            t.setEventHandler { [weak self] in
                self?.tick()
            }
            t.resume()
            timer = t
        } else {
            timer?.cancel()
            timer = nil
        }
    }

    /// A 360 shot was just sent, with the phone at `position`.
    func shotSent(at position: SIMD3<Float>) {
        lock.lock()
        pending = (position, ProcessInfo.processInfo.systemUptime)
        lock.unlock()
        queue.async { [weak self] in
            self?.tick()
        }
    }

    /// The shot in flight ended: `shot` is where it was taken and whether
    /// the export will keep it, nil if no shot was taken.
    func shotFinished(_ shot: Shot?) {
        lock.lock()
        pending = nil
        if let shot {
            shots.append(shot)
            shotsVersion += 1
        }
        lock.unlock()
        queue.async { [weak self] in
            self?.tick()
        }
    }

    /// The latest plan. Cheap; safe on any thread.
    func snapshot() -> Snapshot {
        lock.lock()
        defer { lock.unlock() }
        return latest
    }

    /// The overlay's directions for the phone at `phone`.
    static func hint(
        for s: Snapshot, phone: simd_float4x4?
    ) -> PanoramaHint {
        let complete = s.spots.filter(\.isComplete).count
        var detail = "Spots: \(s.spots.count) (\(complete) complete)"
            + " · 360 shots: \(s.shotCount)"
        if s.movedCount > 0 {
            detail += ", \(s.movedCount) moved"
        }
        if s.status != .needMap {
            let pct = Int((s.coveredFraction * 100).rounded())
            detail += " · \(pct)% seen from 2+ spots"
        }
        func say(_ text: String, arrow: Double? = nil) -> PanoramaHint {
            PanoramaHint(
                instruction: text, detail: detail, arrowDegrees: arrow)
        }
        if s.pending != nil {
            return say("360 shot in progress, hold still")
        }
        guard let target = s.target else {
            switch s.status {
            case .needMap:
                return say("Look around so the LiDAR maps the room")
            case .covered:
                return say("Room covered by 360 shots")
            case .planning:
                return say("Finding the next spot…")
            }
        }
        guard let phone else {
            return say("Walk to the blue globes")
        }
        let p = SIMD3<Float>(
            phone.columns.3.x, phone.columns.3.y, phone.columns.3.z)
        // x, z from the phone to the spot.
        let toSpot = target.center - SIMD2(p.x, p.z)
        let distance = simd_length(toSpot)
        if distance > arriveRadius {
            // Angles clockwise from -z (ahead, seen from above).
            let ahead = SIMD2<Float>(-phone.columns.2.x, -phone.columns.2.z)
            var deg = Double(
                atan2f(toSpot.x, -toSpot.y) - atan2f(ahead.x, -ahead.y)
            ) * 180 / Double.pi
            while deg > 180 { deg -= 360 }
            while deg <= -180 { deg += 360 }
            let what = s.targetIsNew ? "Next spot" : "Finish this spot"
            return say(
                String(format: "%@: %.1f m", what, Double(distance)),
                arrow: deg)
        }
        guard let level = target.nextLevel else {
            return say("Spot done, finding the next one…")
        }
        let name = levelNames[level].capitalized
        let dy = target.levelY[level] - p.y
        if abs(dy) > 0.2 {
            let way = dy > 0 ? "raise" : "lower"
            return say(String(
                format: "%@ shot: %@ the camera %.0f cm",
                name, way, Double(abs(dy) * 100)))
        }
        return say("\(name) shot: hold still at the globe")
    }

    // MARK: - Planning (queue)

    /// Replans when the shots changed or the map grew (by 5%, or at all
    /// after 8 s), then picks the target for where the phone is now.
    private func tick() {
        let now = ProcessInfo.processInfo.systemUptime
        lock.lock()
        let active = timer != nil
        let coverage = self.coverage
        let trajectory = self.trajectory
        let shots = self.shots
        let shotsVersion = self.shotsVersion
        let gen = generation
        if let p = pending, now - p.since > 30 {
            // A shot whose end never came.
            pending = nil
        }
        let pendingPosition = pending?.position
        lock.unlock()
        guard active, let coverage else { return }

        let cov = coverage.snapshot()
        let cellsChanged = cov.version != plannedCoverageVersion
        let grew = abs(cov.count - plannedCellCount)
            >= max(300, plannedCellCount / 20)
        if plan == nil || shotsVersion != plannedShotsVersion
            || (cellsChanged && (grew || now - plannedAt > 8))
        {
            plan = makePlan(
                cov, shots: shots,
                trajectory: trajectory?.positions(every: 10) ?? [])
            plannedCoverageVersion = cov.version
            plannedCellCount = cov.count
            plannedShotsVersion = shotsVersion
            plannedAt = now
        }
        guard let plan else { return }
        let phone = trajectory?.latestPose().map {
            SIMD2<Float>($0.columns.3.x, $0.columns.3.z)
        }
        let (target, isNew) = chooseTarget(plan, phone: phone)
        publish(
            plan, target: target, isNew: isNew,
            pending: pendingPosition, generation: gen)
    }

    private func publish(
        _ plan: Plan, target: SpotMarker?, isNew: Bool,
        pending: SIMD3<Float>?, generation gen: Int
    ) {
        let dotsChanged = plan.id != publishedPlanID
        guard dotsChanged || target != publishedTarget
            || isNew != publishedTargetIsNew || pending != publishedPending
        else { return }
        publishedPlanID = plan.id
        publishedTarget = target
        publishedTargetIsNew = isNew
        publishedPending = pending
        lock.lock()
        defer { lock.unlock() }
        guard gen == generation else { return }
        let version = nextVersion
        nextVersion += 1
        latest = Snapshot(
            version: version,
            dotsVersion: dotsChanged ? version : latest.dotsVersion,
            status: plan.status, floorY: plan.floorY, spots: plan.spots,
            target: target, targetIsNew: isNew, pending: pending,
            shotCount: plan.shotCount, movedCount: plan.movedCount,
            coveredFraction: plan.coveredFraction,
            dotCount: plan.dotCount, positions: plan.positions,
            colors: plan.colors, indices: plan.indices)
    }

    /// An unfinished spot near the phone, else the best new spot: most
    /// gain, nearer first. The spot already shown stays unless another
    /// scores 25% better, so the target doesn't jump while walking.
    private func chooseTarget(
        _ plan: Plan, phone: SIMD2<Float>?
    ) -> (SpotMarker?, Bool) {
        if let phone {
            var nearest: SpotMarker?
            var nearestDistance = Self.finishRadius
            for spot in plan.spots where !spot.isComplete {
                let d = simd_distance(spot.center, phone)
                if d <= nearestDistance {
                    nearest = spot
                    nearestDistance = d
                }
            }
            if let nearest {
                previousTarget = nil
                return (nearest, false)
            }
        }
        guard plan.status == .planning, !plan.candidates.isEmpty else {
            previousTarget = nil
            return (nil, false)
        }
        func score(_ c: Candidate) -> Float {
            let d = phone.map { simd_distance(c.center, $0) } ?? 0
            return c.gain / (1 + 0.25 * d)
        }
        var best = plan.candidates[0]
        for c in plan.candidates.dropFirst() where score(c) > score(best) {
            best = c
        }
        if let previous = previousTarget,
           let kept = plan.candidates.first(where: {
               simd_distance($0.center, previous) < 0.2
           }),
           score(kept) >= 0.8 * score(best)
        {
            best = kept
        }
        previousTarget = best.center
        let states = best.blocked.map {
            $0 ? LevelState.blocked : LevelState.todo
        }
        return (
            SpotMarker(
                center: best.center, levelY: best.levelY, states: states),
            true)
    }

    /// Floor, spots, what the spots see, and the free places to try next
    /// with their gain. Port of pano_levels_sim.py.
    private func makePlan(
        _ cov: CoverageMap.Snapshot, shots: [Shot],
        trajectory: [SIMD3<Float>]
    ) -> Plan {
        planCounter += 1
        let id = planCounter
        let moved = shots.filter { !$0.good }.count
        func bare(_ status: Status, floorY: Float? = nil) -> Plan {
            Plan(
                id: id, status: status, floorY: floorY, spots: [],
                candidates: [], shotCount: shots.count, movedCount: moved,
                coveredFraction: 0, dotCount: 0, positions: Data(),
                colors: Data(), indices: Data())
        }

        // The map's 5 cm cells.
        let n5 = cov.count
        guard n5 >= 200, cov.positions.count >= 12 * n5 else {
            return bare(.needMap)
        }
        var cells5 = [SIMD3<Float>](repeating: .zero, count: n5)
        cov.positions.withUnsafeBytes { raw in
            let f = raw.bindMemory(to: Float.self)
            for i in 0..<n5 {
                cells5[i] = SIMD3(f[3 * i], f[3 * i + 1], f[3 * i + 2])
            }
        }

        // The floor: below the phone's usual height.
        let heights = trajectory.map(\.y).sorted()
        guard !heights.isEmpty,
              let floorY = Self.findFloor(
                  cells5, below: heights[heights.count / 2] - 0.6)
        else { return bare(.needMap) }

        // Planning cells.
        var members: [Int64: Int32] = [:]
        members.reserveCapacity(n5 / 2)
        for p in cells5 {
            members[Self.key(p, cell: Self.planCell), default: 0] += 1
        }
        var cells: [SIMD3<Float>] = []
        cells.reserveCapacity(members.count)
        for (k, m) in members where m >= Int32(Self.minMembers) {
            cells.append(Self.centre(of: k, cell: Self.planCell))
        }
        let nc = cells.count
        guard nc > 0 else { return bare(.needMap, floorY: floorY) }

        // The levels' heights: the user's own good shots, else defaults.
        var levelHeights = Self.defaultLevelHeights
        for l in 0..<3 {
            let hs = shots.compactMap { s -> Float? in
                let h = s.position.y - floorY
                return s.good && Self.level(forHeight: h) == l ? h : nil
            }.sorted()
            if !hs.isEmpty {
                levelHeights[l] = hs[hs.count / 2]
            }
        }
        let levelY = levelHeights.map { floorY + $0 }
        // Cells below this don't block a shot at level l: the floor, and
        // for the low shot the floor's LiDAR noise.
        func ignoreBelow(_ l: Int) -> Float {
            floorY + (l == 2 ? Self.lowShotIgnore : Self.floorBand)
        }
        var occupied = Set<Int64>(minimumCapacity: n5)
        for p in cells5 {
            occupied.insert(Self.key(p, cell: CoverageMap.cellSize))
        }

        // Spots: shots grouped by position, levels by height.
        var groups: [(center: SIMD2<Float>, members: [Shot])] = []
        for s in shots {
            let xz = SIMD2<Float>(s.position.x, s.position.z)
            var join = -1
            var joinDistance = Self.spotRadius
            for (i, group) in groups.enumerated() {
                let d = simd_distance(group.center, xz)
                if d < joinDistance {
                    join = i
                    joinDistance = d
                }
            }
            if join >= 0 {
                groups[join].members.append(s)
                var sum = SIMD2<Float>.zero
                for m in groups[join].members {
                    sum += SIMD2<Float>(m.position.x, m.position.z)
                }
                groups[join].center = sum / Float(groups[join].members.count)
            } else {
                groups.append((center: xz, members: [s]))
            }
        }
        var spots: [SpotMarker] = []
        for group in groups {
            var states = [LevelState](repeating: .todo, count: 3)
            for s in group.members {
                let l = Self.level(forHeight: s.position.y - floorY)
                if s.good {
                    states[l] = .done
                } else if states[l] != .done {
                    states[l] = .failed
                }
            }
            for l in 0..<3 where states[l] == .todo {
                let p = SIMD3<Float>(group.center.x, levelY[l], group.center.y)
                if !Self.isClear(
                    p, ignoreBelow: ignoreBelow(l), occupied: occupied)
                {
                    states[l] = .blocked
                }
            }
            spots.append(SpotMarker(
                center: group.center, levelY: levelY, states: states))
        }

        // What the spots see: a cell counts once per spot.
        var spotCount = [Int32](repeating: 0, count: nc)
        var reach = [Bool](repeating: false, count: nc)
        var seen = [Bool](repeating: false, count: nc)
        for group in groups {
            let good = group.members.filter(\.good)
            guard !good.isEmpty else { continue }
            for i in 0..<nc { seen[i] = false }
            for s in good {
                visibleDistances(from: s.position, cells: cells)
                let vis = visible
                for i in 0..<nc where vis[i].isFinite { seen[i] = true }
            }
            for i in 0..<nc where seen[i] {
                spotCount[i] += 1
                reach[i] = true
            }
        }

        // Where one can stand: seen floor with nothing above it (one-cell
        // gaps bridged), joined to where the phone has been.
        var minX = Float.infinity, maxX = -Float.infinity
        var minZ = Float.infinity, maxZ = -Float.infinity
        for p in cells5 {
            minX = min(minX, p.x)
            maxX = max(maxX, p.x)
            minZ = min(minZ, p.z)
            maxZ = max(maxZ, p.z)
        }
        let gs = Self.planCell
        let x0 = minX - 0.5, z0 = minZ - 0.5
        let nx = Int((maxX + 0.5 - x0) / gs) + 1
        let nz = Int((maxZ + 0.5 - z0) / gs) + 1
        guard nx > 0, nz > 0, nx * nz <= 4_000_000 else {
            return bare(.needMap, floorY: floorY)
        }
        func column(_ x: Float, _ z: Float) -> (Int, Int) {
            (Int(((x - x0) / gs).rounded(.down)),
             Int(((z - z0) / gs).rounded(.down)))
        }
        var floorColumn = [Bool](repeating: false, count: nx * nz)
        var blockedColumn = [Bool](repeating: false, count: nx * nz)
        for p in cells5 {
            let (i, j) = column(p.x, p.z)
            guard i >= 0, i < nx, j >= 0, j < nz else { continue }
            let h = p.y - floorY
            if abs(h) < Self.floorBand {
                floorColumn[i * nz + j] = true
            }
            if h > Self.floorNoiseTop && h < Self.standTop {
                blockedColumn[i * nz + j] = true
            }
        }
        func standable(_ i: Int, _ j: Int) -> Bool {
            i >= 0 && i < nx && j >= 0 && j < nz
                && floorColumn[i * nz + j] && !blockedColumn[i * nz + j]
        }
        var walk = [Bool](repeating: false, count: nx * nz)
        for i in 0..<nx {
            for j in 0..<nz where !blockedColumn[i * nz + j] {
                walk[i * nz + j] = standable(i, j)
                    || standable(i - 1, j) || standable(i + 1, j)
                    || standable(i, j - 1) || standable(i, j + 1)
            }
        }
        var reachable = [Bool](repeating: false, count: nx * nz)
        var stack: [Int] = []
        func visit(_ i: Int, _ j: Int) {
            guard i >= 0, i < nx, j >= 0, j < nz else { return }
            let k = i * nz + j
            if walk[k] && !reachable[k] {
                reachable[k] = true
                stack.append(k)
            }
        }
        for p in trajectory {
            let (i, j) = column(p.x, p.z)
            for di in -3...3 {
                for dj in -3...3 {
                    visit(i + di, j + dj)
                }
            }
        }
        while let k = stack.popLast() {
            let i = k / nz, j = k % nz
            visit(i + 1, j)
            visit(i - 1, j)
            visit(i, j + 1)
            visit(i, j - 1)
        }
        var reachableCount = 0
        for r in reachable where r { reachableCount += 1 }

        // Free places to stand next: on a lattice over the reachable
        // floor, away from spots shot already, camera clear at chest
        // height; each with the levels that are clear.
        let step = max(
            Self.lattice,
            (Float(reachableCount) * gs * gs / Self.maxCandidates)
                .squareRoot())
        let shotCenters = spots
            .filter { $0.states.contains(.done) }
            .map(\.center)
        var candidates: [Candidate] = []
        var x = x0 + 0.5
        while x < x0 + Float(nx) * gs - 0.5 {
            var z = z0 + 0.5
            while z < z0 + Float(nz) * gs - 0.5 {
                defer { z += step }
                let (i, j) = column(x, z)
                var onFloor = false
                for di in -1...1 {
                    for dj in -1...1 {
                        let a = i + di, b = j + dj
                        if a >= 0, a < nx, b >= 0, b < nz,
                           reachable[a * nz + b]
                        {
                            onFloor = true
                        }
                    }
                }
                guard onFloor else { continue }
                let xz = SIMD2<Float>(x, z)
                guard !shotCenters.contains(where: {
                    simd_distance($0, xz) < Self.spotSpacing
                }) else { continue }
                guard Self.isClear(
                    SIMD3<Float>(x, levelY[1], z),
                    ignoreBelow: ignoreBelow(1), occupied: occupied)
                else { continue }
                let blocked = (0..<3).map { l in
                    !Self.isClear(
                        SIMD3<Float>(x, levelY[l], z),
                        ignoreBelow: ignoreBelow(l), occupied: occupied)
                }
                candidates.append(Candidate(
                    center: xz, levelY: levelY, blocked: blocked, gain: 0))
            }
            x += step
        }

        // Gain of each place: the cells its shots would see, more for
        // cells no spot sees yet, less for far ones.
        var minDistance = [Float](repeating: .infinity, count: nc)
        let fullW = Self.fullWeightDistance
        let fallSpan = Self.maxView - Self.fullWeightDistance
        for c in candidates.indices {
            for i in 0..<nc { minDistance[i] = .infinity }
            for l in 0..<3 where !candidates[c].blocked[l] {
                let eye = SIMD3<Float>(
                    candidates[c].center.x, candidates[c].levelY[l],
                    candidates[c].center.y)
                visibleDistances(from: eye, cells: cells)
                let vis = visible
                for i in 0..<nc where vis[i] < minDistance[i] {
                    minDistance[i] = vis[i]
                }
            }
            var gain: Float = 0
            for i in 0..<nc {
                let r = minDistance[i]
                guard r.isFinite else { continue }
                reach[i] = true
                let k = spotCount[i]
                guard k < 2 else { continue }
                let w: Float = r <= fullW
                    ? 1 : max(0.3, 1 - (r - fullW) / fallSpan * 0.7)
                gain += (k == 0 ? 1 : 0.5) * w
            }
            candidates[c].gain = gain
        }

        var reachCount = 0
        var twice = 0
        for i in 0..<nc where reach[i] {
            reachCount += 1
            if spotCount[i] >= 2 { twice += 1 }
        }
        let bestGain = candidates.map(\.gain).max() ?? 0
        let status: Status
        if candidates.isEmpty && spots.isEmpty {
            status = .needMap
        } else if candidates.isEmpty
            || bestGain < max(15, 0.02 * Float(reachCount))
        {
            status = .covered
        } else {
            status = .planning
        }

        // The planning cells as dots.
        var positions = [Float]()
        positions.reserveCapacity(3 * nc)
        var colors = [Float]()
        colors.reserveCapacity(3 * nc)
        for i in 0..<nc {
            let p = cells[i]
            positions.append(p.x)
            positions.append(p.y)
            positions.append(p.z)
            let color: SIMD3<Float>
            if spotCount[i] >= 2 {
                color = Self.seenTwiceColor
            } else if spotCount[i] == 1 {
                color = Self.seenOnceColor
            } else if reach[i] {
                color = Self.unseenColor
            } else {
                color = Self.unreachableColor
            }
            colors.append(color.x)
            colors.append(color.y)
            colors.append(color.z)
        }
        let indices = [UInt32](0..<UInt32(nc))
        return Plan(
            id: id, status: status, floorY: floorY, spots: spots,
            candidates: candidates.filter { $0.gain > 0 },
            shotCount: shots.count, movedCount: moved,
            coveredFraction: reachCount > 0
                ? Float(twice) / Float(reachCount) : 0,
            dotCount: nc,
            positions: positions.withUnsafeBufferPointer { Data(buffer: $0) },
            colors: colors.withUnsafeBufferPointer { Data(buffer: $0) },
            indices: indices.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    // MARK: - Helpers

    private static let keyOffset: Int64 = 1 << 20

    /// Cell key of `p` for cells of edge `cell`, packed as CoverageMap
    /// packs its keys (21 bits per axis).
    private static func key(_ p: SIMD3<Float>, cell: Float) -> Int64 {
        let c = (p / cell).rounded(.down)
        return ((Int64(c.x) + keyOffset) << 42)
            | ((Int64(c.y) + keyOffset) << 21)
            | (Int64(c.z) + keyOffset)
    }

    private static func centre(of key: Int64, cell: Float) -> SIMD3<Float> {
        let m: Int64 = (1 << 21) - 1
        let x = Float(((key >> 42) & m) - keyOffset)
        let y = Float(((key >> 21) & m) - keyOffset)
        let z = Float((key & m) - keyOffset)
        return (SIMD3<Float>(x, y, z) + 0.5) * cell
    }

    /// The floor's height: the lowest 5 cm height band below `top` with at
    /// least half the cells of the fullest one (furniture tops are higher;
    /// the floor is the lowest large surface), refined to the mean height
    /// of the cells within 10 cm of it. Nil without enough low cells.
    private static func findFloor(
        _ cells5: [SIMD3<Float>], below top: Float
    ) -> Float? {
        let s = CoverageMap.cellSize
        var bins: [Int: Int] = [:]
        for p in cells5 where p.y < top {
            bins[Int((p.y / s).rounded(.down)), default: 0] += 1
        }
        guard let most = bins.values.max(), most >= 50,
              let k = bins.filter({ $0.value * 2 >= most }).keys.min()
        else { return nil }
        let middle = (Float(k) + 0.5) * s
        var sum: Float = 0
        var n = 0
        for p in cells5 where p.y < top && abs(p.y - middle) <= 0.1 {
            sum += p.y
            n += 1
        }
        return n > 0 ? sum / Float(n) : nil
    }

    /// No 5 cm cell higher than `floorLimit` within clearRadius of `p`.
    private static func isClear(
        _ p: SIMD3<Float>, ignoreBelow floorLimit: Float,
        occupied: Set<Int64>
    ) -> Bool {
        let s = CoverageMap.cellSize
        let n = Int((clearRadius / s).rounded(.up))
        let c = (p / s).rounded(.down)
        let cx = Int64(c.x), cy = Int64(c.y), cz = Int64(c.z)
        let r2 = clearRadius * clearRadius
        for dx in -n...n {
            for dy in -n...n {
                for dz in -n...n {
                    let kx = cx + Int64(dx)
                    let ky = cy + Int64(dy)
                    let kz = cz + Int64(dz)
                    let centre = (SIMD3<Float>(
                        Float(kx), Float(ky), Float(kz)) + 0.5) * s
                    guard centre.y > floorLimit,
                          simd_distance_squared(centre, p) < r2
                    else { continue }
                    let key = ((kx + keyOffset) << 42)
                        | ((ky + keyOffset) << 21)
                        | (kz + keyOffset)
                    if occupied.contains(key) {
                        return false
                    }
                }
            }
        }
        return true
    }

    /// Fills `visible` with the distance from `eye` to each planning cell
    /// a 360 photo there would see, .infinity for the others: within
    /// maxView, and not more than 0.1 m + 5% behind the nearest cell in its
    /// direction. The cells are drawn into an equirect depth map (1.4° per
    /// pixel), each as a disc of its own angular size, then tested.
    private func visibleDistances(
        from eye: SIMD3<Float>, cells: [SIMD3<Float>]
    ) {
        let n = cells.count
        if visible.count != n {
            visible = [Float](repeating: .infinity, count: n)
        }
        let w = Self.mapWidth, h = Self.mapHeight
        let fw = Float(w), fh = Float(h)
        let maxR2 = Self.maxView * Self.maxView
        let halfCell = 0.5 * Self.planCell
        projected.removeAll(keepingCapacity: true)
        depthMap.withUnsafeMutableBufferPointer { depth in
            for k in 0..<depth.count {
                depth[k] = .infinity
            }
            for i in 0..<n {
                let d = cells[i] - eye
                let r2 = simd_length_squared(d)
                guard r2 > 0.0225, r2 <= maxR2 else { continue }
                let r = r2.squareRoot()
                let lat = asinf(max(-1, min(1, d.y / r)))
                let lon = atan2f(d.x, -d.z)
                let u = (lon + Float.pi) * (fw / (2 * Float.pi))
                let v = (Float.pi / 2 - lat) * (fh / Float.pi)
                let rv = (halfCell / r) * (fh / Float.pi)
                let ru = rv / max(cosf(lat), 1e-3)
                let v0 = max(0, min(h - 1, Int((v - rv).rounded(.down))))
                let v1 = max(0, min(h - 1, Int((v + rv).rounded(.down))))
                var u0 = Int((u - ru).rounded(.down))
                var nu = Int((u + ru).rounded(.down)) - u0 + 1
                if nu >= w {
                    u0 = 0
                    nu = w
                }
                for row in v0...v1 {
                    let base = row * w
                    for k in 0..<nu {
                        var col = (u0 + k) % w
                        if col < 0 { col += w }
                        if r < depth[base + col] {
                            depth[base + col] = r
                        }
                    }
                }
                projected.append(
                    Projected(index: Int32(i), u: u, v: v, r: r))
            }
        }
        for i in 0..<n {
            visible[i] = .infinity
        }
        depthMap.withUnsafeBufferPointer { depth in
            for p in projected {
                var pu = Int(p.u.rounded(.down)) % w
                if pu < 0 { pu += w }
                let pv = max(0, min(h - 1, Int(p.v.rounded(.down))))
                if p.r <= depth[pv * w + pu] + 0.10 + 0.05 * p.r {
                    visible[Int(p.index)] = p.r
                }
            }
        }
    }
}
