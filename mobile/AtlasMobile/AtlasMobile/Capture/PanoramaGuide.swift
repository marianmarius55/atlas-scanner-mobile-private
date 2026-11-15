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
/// - the room is shot in three passes: first the high shot everywhere,
///   then the chest shot at every spot of the high pass, then the low shot
///   at those spots. A pass ends for good once it's done, so the guide
///   doesn't go back to an earlier height;
/// - the high pass places its spots by the route chosen in Camera
///   Settings (Route): coverage, where a cell counts as covered once 2
///   spots' high shots see it and the next spot is the nearest free place
///   that adds at least half as much as the best one; or loops, along the
///   walls and doorways first, then around the furniture and through the
///   middle, the set spacing apart. In the chest and low passes the next
///   spot is the nearest one still missing that height.
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
    // MARK: - Settings

    /// How the high pass places its spots (Camera Settings, 360 Guide).
    enum Route: String, CaseIterable, Identifiable {
        /// Wherever the most of the room not yet seen twice is seen from.
        case coverage
        /// A loop along the walls, one around the furniture and through the
        /// middle, and three spots at each doorway.
        case loops

        var id: String { rawValue }

        var title: String {
            switch self {
            case .coverage: return "Coverage (original)"
            case .loops: return "Loops: walls, furniture, doorways"
            }
        }

        var detail: String {
            switch self {
            case .coverage:
                return "Each next spot is the free place that sees the most of"
                    + " the room not yet seen from 2 spots. Spacing: the least"
                    + " distance between spots."
            case .loops:
                return "First a loop along the walls, about 0.65 m from them"
                    + " (the middle of narrower floor), with 3 spots at each"
                    + " doorway (before, in, after); then"
                    + " a loop around the furniture and through the middle."
                    + " Spacing: the distance between spots along a loop."
            }
        }

        var defaultSpacing: Double { self == .coverage ? 0.6 : 1.0 }
        var spacingKey: String { "panoramaGuideSpacing_\(rawValue)" }
    }

    static let routeKey = "panoramaGuideRoute"
    static let spacingRange: ClosedRange<Double> = 0.4...2.0

    struct Settings: Equatable {
        let route: Route
        /// Metres between spots.
        let spacing: Float
    }

    /// The current settings; read from any thread.
    static var settings: Settings {
        let d = UserDefaults.standard
        let route = d.string(forKey: routeKey)
            .flatMap(Route.init(rawValue:)) ?? .coverage
        let stored = d.object(forKey: route.spacingKey) as? Double
        let spacing = min(max(stored ?? route.defaultSpacing,
                              spacingRange.lowerBound),
                          spacingRange.upperBound)
        return Settings(route: route, spacing: Float(spacing))
    }

    // MARK: - Tuning

    /// Loops: distance from the walls (and anything tall) of the wall loop.
    static let wallBand: ClosedRange<Float> = 0.5...0.8
    /// Loops: distance from the furniture of the furniture loop.
    static let furnitureBand: ClosedRange<Float> = 0.4...0.8
    /// Loops: an obstacle with LiDAR cells at these heights above the
    /// floor is a wall (or a wardrobe); lower ones are furniture. It stops
    /// below the ceiling, which is over every column of the room.
    static let wallHeights: ClosedRange<Float> = 1.7...2.1
    /// Loops: the spots before and after a doorway are this far from it.
    static let doorwayStep: Float = 0.7

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
        /// The target's level in the current pass; nil for spots.
        var focusLevel: Int? = nil

        var isComplete: Bool {
            !states.contains(where: { $0 == .todo || $0 == .failed })
        }

        /// The level to shoot here: the pass's for the target, else high,
        /// then chest, then low.
        var nextLevel: Int? {
            if let l = focusLevel {
                return states[l] == .todo || states[l] == .failed ? l : nil
            }
            return states.firstIndex(where: { $0 == .todo || $0 == .failed })
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
        /// The pass: 0 high, 1 chest, 2 low, 3 all done.
        let pass: Int
        /// How the high pass places its spots.
        let route: Route
        /// High pass with the coverage route: share of the cells free
        /// places can see that 2+ spots' high shots see. Otherwise: share
        /// of the spots done.
        let coveredFraction: Float
        /// Spots done at the pass's height, and in all (the coverage
        /// route's high pass: unused).
        let passDone: Int
        let passTotal: Int
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
                pending: nil, shotCount: 0, movedCount: 0, pass: 0,
                route: .coverage, coveredFraction: 0, passDone: 0,
                passTotal: 0, dotCount: 0,
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
        /// Loops: 0 the wall loop and doorways, 1 the furniture loop and
        /// the middle, shot after stage 0. Coverage: always 0.
        var stage: Int = 0
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
        let pass: Int
        let route: Route
        /// This pass has nothing left to do.
        let finished: Bool
        let coveredFraction: Float
        let passDone: Int
        let passTotal: Int
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
    /// The pass reached: 0 high, 1 chest, 2 low, 3 all done. Only goes up.
    private var pass = 0
    private var plannedSettings: Settings?
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
            pass = 0
            plannedSettings = nil
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
        var detail = s.pass < 3
            ? "\(levelNames[s.pass].capitalized) pass (\(s.pass + 1)/3)"
            : "All 3 passes done"
        detail += " · 360 shots: \(s.shotCount)"
        if s.movedCount > 0 {
            detail += ", \(s.movedCount) moved"
        }
        if s.status != .needMap && s.pass == 0 && s.route == .coverage {
            let pct = Int((s.coveredFraction * 100).rounded())
            detail += " · \(pct)% seen from 2+ spots"
        } else if s.status != .needMap && s.pass < 3 {
            detail += " · \(s.passDone)/\(s.passTotal) spots done"
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
            let what = target.focusLevel.map {
                "Nearest \(levelNames[$0]) shot"
            } ?? "Next spot"
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
        let settings = Self.settings
        if plan == nil || shotsVersion != plannedShotsVersion
            || settings != plannedSettings
            || (cellsChanged && (grew || now - plannedAt > 8))
        {
            let path = trajectory?.positions(every: 10) ?? []
            var made = makePlan(
                cov, shots: shots, trajectory: path, pass: pass,
                settings: settings)
            // A finished pass moves on to the next height for good.
            while made.finished && pass < 3 {
                pass += 1
                previousTarget = nil
                made = makePlan(
                    cov, shots: shots, trajectory: path, pass: pass,
                    settings: settings)
            }
            plan = made
            plannedSettings = settings
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
            pass: plan.pass, route: plan.route,
            coveredFraction: plan.coveredFraction,
            passDone: plan.passDone, passTotal: plan.passTotal,
            dotCount: plan.dotCount, positions: plan.positions,
            colors: plan.colors, indices: plan.indices)
    }

    /// The globe nearest the phone that still needs this pass's height.
    /// High pass: a spot whose high shot is missing or failed, or a free
    /// place adding at least half as much as the best one (loops: a place
    /// of the route, the wall loop's first). Chest and low
    /// pass: a spot of the high pass missing that height. The globe shown
    /// stays unless another is 0.5 m nearer, so it doesn't jump while
    /// walking.
    private func chooseTarget(
        _ plan: Plan, phone: SIMD2<Float>?
    ) -> (SpotMarker?, Bool) {
        let l = plan.pass
        guard plan.status == .planning, l < 3 else {
            previousTarget = nil
            return (nil, false)
        }
        var options: [(marker: SpotMarker, isNew: Bool, gain: Float)] = []
        for spot in plan.spots {
            let s = spot.states[l]
            guard s == .todo || s == .failed else { continue }
            if l > 0 && spot.states[0] != .done { continue }
            var marker = spot
            marker.focusLevel = l
            options.append((marker, false, .infinity))
        }
        if l == 0 {
            // Loops: the wall loop and doorways before the furniture loop.
            let stage = plan.candidates.map(\.stage).min() ?? 0
            let best = plan.candidates.map(\.gain).max() ?? 0
            for c in plan.candidates
            where c.stage == stage && !c.blocked[0]
                && c.gain >= 0.5 * best {
                let states = c.blocked.map {
                    $0 ? LevelState.blocked : LevelState.todo
                }
                options.append((
                    SpotMarker(center: c.center, levelY: c.levelY,
                               states: states, focusLevel: 0),
                    true, c.gain))
            }
        }
        guard !options.isEmpty else {
            previousTarget = nil
            return (nil, false)
        }
        guard let phone else {
            // No pose yet: the spot needing it, else the most gain.
            let pick = options.max { $0.gain < $1.gain }!
            previousTarget = pick.marker.center
            return (pick.marker, pick.isNew)
        }
        func distance(_ o: (marker: SpotMarker, isNew: Bool, gain: Float))
            -> Float {
            simd_distance(o.marker.center, phone)
        }
        var pick = options.min { distance($0) < distance($1) }!
        if let previous = previousTarget,
           let kept = options.first(where: {
               simd_distance($0.marker.center, previous) < 0.2
           }),
           distance(kept) <= distance(pick) + 0.5
        {
            pick = kept
        }
        previousTarget = pick.marker.center
        return (pick.marker, pick.isNew)
    }

    /// Floor, spots, what the spots see, and the free places to try next
    /// with their gain. Port of pano_levels_sim.py.
    private func makePlan(
        _ cov: CoverageMap.Snapshot, shots: [Shot],
        trajectory: [SIMD3<Float>], pass: Int, settings: Settings
    ) -> Plan {
        planCounter += 1
        let id = planCounter
        let moved = shots.filter { !$0.good }.count
        // The height whose shots count: the pass's, the low one when done.
        let passLevel = min(pass, 2)
        func bare(_ status: Status, floorY: Float? = nil) -> Plan {
            Plan(
                id: id, status: status, floorY: floorY, spots: [],
                candidates: [], shotCount: shots.count, movedCount: moved,
                pass: pass, route: settings.route, finished: false,
                coveredFraction: 0,
                passDone: 0, passTotal: 0, dotCount: 0, positions: Data(),
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
            // Only this pass's height: the dots show what it has covered.
            let good = group.members.filter {
                $0.good && Self.level(forHeight: $0.position.y - floorY)
                    == passLevel
            }
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
        // Something at wallHeights: a wall or a wardrobe, not furniture.
        var tallColumn = [Bool](repeating: false, count: nx * nz)
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
            if Self.wallHeights.contains(h) {
                tallColumn[i * nz + j] = true
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
        // Only the high pass places new spots; chest and low go back to
        // its spots.
        let shotCenters = spots.map(\.center)
        var candidates: [Candidate] = []
        var x = x0 + 0.5
        while pass == 0, settings.route == .coverage,
              x < x0 + Float(nx) * gs - 0.5 {
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
                    simd_distance($0, xz) < settings.spacing
                }) else { continue }
                guard Self.isClear(
                    SIMD3<Float>(x, levelY[0], z),
                    ignoreBelow: ignoreBelow(0), occupied: occupied)
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

        // Loops: the route's places, around the spots shot already.
        if pass == 0 && settings.route == .loops {
            // Furniture: something in the way that isn't tall. Walls:
            // anything else one can't stand on (tall things, and past
            // the mapped floor).
            var furniture = [Bool](repeating: false, count: nx * nz)
            var wall = [Bool](repeating: false, count: nx * nz)
            for k in 0..<(nx * nz) {
                furniture[k] = blockedColumn[k] && !tallColumn[k]
                wall[k] = !reachable[k] && !furniture[k]
            }
            let start = trajectory.first.map { SIMD2<Float>($0.x, $0.z) }
            candidates = Self.loopCandidates(
                LoopGrid(nx: nx, nz: nz, x0: x0, z0: z0, gs: gs,
                         reachable: reachable, wall: wall,
                         furniture: furniture),
                spacing: settings.spacing, start: start,
                taken: shotCenters
            ) { xz in
                // A place is usable when the high shot is clear there.
                guard Self.isClear(
                    SIMD3<Float>(xz.x, levelY[0], xz.y),
                    ignoreBelow: ignoreBelow(0), occupied: occupied)
                else { return nil }
                return (0..<3).map { l in
                    !Self.isClear(
                        SIMD3<Float>(xz.x, levelY[l], xz.y),
                        ignoreBelow: ignoreBelow(l), occupied: occupied)
                }
            }.map {
                Candidate(center: $0.center, levelY: levelY,
                          blocked: $0.blocked, gain: 1, stage: $0.stage)
            }
        }

        // Gain of each place: the cells its shots would see, more for
        // cells no spot sees yet, less for far ones. The loops route
        // places by the route alone.
        var minDistance = [Float](repeating: .infinity, count: nc)
        let fullW = Self.fullWeightDistance
        let fallSpan = Self.maxView - Self.fullWeightDistance
        for c in candidates.indices where settings.route == .coverage {
            for i in 0..<nc { minDistance[i] = .infinity }
            // The high pass: what the high shot would see.
            for l in 0..<1 where !candidates[c].blocked[l] {
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

        if settings.route == .loops {
            // No visibility per place: a cell no spot sees yet shows red.
            for i in 0..<nc { reach[i] = true }
        }
        var reachCount = 0
        var twice = 0
        for i in 0..<nc where reach[i] {
            reachCount += 1
            if spotCount[i] >= 2 { twice += 1 }
        }
        let bestGain = candidates.map(\.gain).max() ?? 0
        // Chest and low pass: the high pass's spots and how many have
        // this height. Loops' high pass: those shot and the route's rest.
        let passSpots = spots.filter {
            $0.states[0] == .done && $0.states[passLevel] != .blocked
        }
        let passDone = passSpots.filter {
            $0.states[passLevel] == .done
        }.count
        let passTotal = pass == 0 && settings.route == .loops
            ? passSpots.count + candidates.count : passSpots.count
        let status: Status
        let finished: Bool
        if pass >= 3 {
            status = .covered
            finished = false
        } else if pass == 0 {
            let missing = spots.contains {
                $0.states[0] == .failed
            }
            if candidates.isEmpty && spots.isEmpty {
                status = .needMap
                finished = false
            } else if !missing && (candidates.isEmpty
                || (settings.route == .coverage
                    && bestGain < max(15, 0.02 * Float(reachCount))))
            {
                status = .covered
                finished = spots.contains { $0.states[0] == .done }
            } else {
                status = .planning
                finished = false
            }
        } else {
            finished = passDone == passSpots.count
            status = finished ? .covered : .planning
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
            pass: pass, route: settings.route, finished: finished,
            coveredFraction: pass == 0 && settings.route == .coverage
                ? (reachCount > 0 ? Float(twice) / Float(reachCount) : 0)
                : (passTotal == 0
                    ? 1 : Float(passDone) / Float(passTotal)),
            passDone: passDone, passTotal: passTotal,
            dotCount: nc,
            positions: positions.withUnsafeBufferPointer { Data(buffer: $0) },
            colors: colors.withUnsafeBufferPointer { Data(buffer: $0) },
            indices: indices.withUnsafeBufferPointer { Data(buffer: $0) })
    }

    // MARK: - Loops route

    /// The floor plan the loops route is laid out on: planCell columns,
    /// column (i, j) at k = i * nz + j, its centre at x0 + (i + 0.5) gs,
    /// z0 + (j + 0.5) gs.
    private struct LoopGrid {
        let nx: Int, nz: Int
        let x0: Float, z0: Float, gs: Float
        /// Floor one can walk on, joined to where the phone has been.
        let reachable: [Bool]
        /// Walls, tall things and everything past the mapped floor.
        let wall: [Bool]
        /// Low things in the way: beds, tables, desks.
        let furniture: [Bool]

        func centre(_ k: Int) -> SIMD2<Float> {
            SIMD2(x0 + (Float(k / nz) + 0.5) * gs,
                  z0 + (Float(k % nz) + 0.5) * gs)
        }

        func index(_ p: SIMD2<Float>) -> Int? {
            let i = Int(((p.x - x0) / gs).rounded(.down))
            let j = Int(((p.y - z0) / gs).rounded(.down))
            guard i >= 0, i < nx, j >= 0, j < nz else { return nil }
            return i * nz + j
        }
    }

    /// The loops route's places that no spot is near yet: the doorways'
    /// and the wall loop's (stage 0), then the furniture loop's and the
    /// middle's (stage 1), `spacing` apart along each loop. `usable` says
    /// whether the high shot is clear at a place (nil when not) and which
    /// levels are blocked there.
    private static func loopCandidates(
        _ g: LoopGrid, spacing s: Float, start: SIMD2<Float>?,
        taken existing: [SIMD2<Float>],
        usable: (SIMD2<Float>) -> [Bool]?
    ) -> [(center: SIMD2<Float>, blocked: [Bool], stage: Int)] {
        let n = g.nx * g.nz
        guard n > 0 else { return [] }
        let toWall = distanceField(g.wall, g)
        let toFurniture = distanceField(g.furniture, g)
        var out: [(center: SIMD2<Float>, blocked: [Bool], stage: Int)] = []
        var taken = existing

        // Doorways: three spots each, before, in and after.
        let doorKeep = min(0.7 * s, 0.5)
        for door in doorways(g, toWall: toWall) {
            for t: Float in [-1, 0, 1] {
                let p = door.centre + t * doorwayStep * door.axis
                guard let k = g.index(p), g.reachable[k],
                      !taken.contains(where: {
                          simd_distance($0, p) < doorKeep
                      }),
                      let blocked = usable(p)
                else { continue }
                taken.append(p)
                out.append((p, blocked, 0))
            }
        }

        // The wall loop: the floor within wallBand's far edge of the walls,
        // best near the band's middle (in a strip narrower than the band,
        // the strip's middle). Then the furniture loop, then the middle.
        // Each place's cost is how far it is from where the loop wants it.
        typealias Band = (points: [SIMD2<Float>], cost: [Float])
        var wallBand: Band = ([], [])
        var furnitureBand: Band = ([], [])
        var middle: Band = ([], [])
        let wallMid = (Self.wallBand.lowerBound + Self.wallBand.upperBound) / 2
        let furnitureMid = (Self.furnitureBand.lowerBound
            + Self.furnitureBand.upperBound) / 2
        for k in 0..<n where g.reachable[k] {
            let w = toWall[k], f = toFurniture[k]
            // Between a wall and furniture closer than twice the band's
            // middle, the floor's middle: w + f is the floor's width there.
            if w <= Self.wallBand.upperBound {
                wallBand.points.append(g.centre(k))
                wallBand.cost.append(abs(w - min(wallMid, (w + f) / 2)))
            } else if f <= Self.furnitureBand.upperBound {
                furnitureBand.points.append(g.centre(k))
                furnitureBand.cost.append(
                    abs(f - min(furnitureMid, (w + f) / 2)))
            } else {
                middle.points.append(g.centre(k))
                middle.cost.append(0)
            }
        }
        var cursor = start
        for (band, stage) in [(wallBand, 0), (furnitureBand, 1),
                              (middle, 1)] {
            let placed = walk(band.points, cost: band.cost, spacing: s,
                              from: cursor, taken: &taken, usable: usable)
            for p in placed {
                out.append((p.center, p.blocked, stage))
            }
            if let last = placed.last { cursor = last.center }
        }
        return out
    }

    /// Distance in metres from each column to the nearest `source` column
    /// (chamfer, 2 passes); .infinity without any.
    private static func distanceField(
        _ source: [Bool], _ g: LoopGrid
    ) -> [Float] {
        let nx = g.nx, nz = g.nz
        let a = g.gs, b = g.gs * Float(2).squareRoot()
        var d = [Float](repeating: .infinity, count: nx * nz)
        for k in 0..<(nx * nz) where source[k] { d[k] = 0 }
        for i in 0..<nx {
            for j in 0..<nz {
                let k = i * nz + j
                var v = d[k]
                if i > 0 {
                    v = min(v, d[k - nz] + a)
                    if j > 0 { v = min(v, d[k - nz - 1] + b) }
                    if j + 1 < nz { v = min(v, d[k - nz + 1] + b) }
                }
                if j > 0 { v = min(v, d[k - 1] + a) }
                d[k] = v
            }
        }
        for i in stride(from: nx - 1, through: 0, by: -1) {
            for j in stride(from: nz - 1, through: 0, by: -1) {
                let k = i * nz + j
                var v = d[k]
                if i + 1 < nx {
                    v = min(v, d[k + nz] + a)
                    if j + 1 < nz { v = min(v, d[k + nz + 1] + b) }
                    if j > 0 { v = min(v, d[k + nz - 1] + b) }
                }
                if j + 1 < nz { v = min(v, d[k + 1] + a) }
                d[k] = v
            }
        }
        return d
    }

    /// Doorways: places where the floor narrows between walls to at most
    /// 1.2 m (walls within 0.8 m on both sides) with open floor 1 m away
    /// on both sides along the passage. Each is the column furthest from
    /// the walls, with the passage's direction.
    private static func doorways(
        _ g: LoopGrid, toWall: [Float]
    ) -> [(centre: SIMD2<Float>, axis: SIMD2<Float>)] {
        let dirs: [SIMD2<Float>] = [
            SIMD2(1, 0), SIMD2(0, 1),
            simd_normalize(SIMD2(1, 1)), simd_normalize(SIMD2(1, -1)),
        ]
        func wallWithin(_ p: SIMD2<Float>, _ dir: SIMD2<Float>,
                        _ reach: Float) -> Bool {
            var t: Float = g.gs
            while t <= reach {
                guard let k = g.index(p + t * dir) else { return true }
                if g.wall[k] { return true }
                t += g.gs * 0.5
            }
            return false
        }
        func open(_ p: SIMD2<Float>) -> Bool {
            guard let k = g.index(p) else { return false }
            return g.reachable[k] && toWall[k] >= 0.7
        }
        var found: [(centre: SIMD2<Float>, axis: SIMD2<Float>, w: Float)] = []
        for k in 0..<(g.nx * g.nz) where g.reachable[k] {
            let w = toWall[k]
            guard w >= 0.25, w <= 0.6 else { continue }
            let p = g.centre(k)
            for n in dirs {
                guard wallWithin(p, n, 0.8), wallWithin(p, -n, 0.8)
                else { continue }
                let axis = SIMD2<Float>(-n.y, n.x)
                guard open(p + axis), open(p - axis) else { continue }
                found.append((p, axis, w))
                break
            }
        }
        // One per doorway: the widest place, others within 0.8 m dropped.
        found.sort { $0.w > $1.w }
        var doors: [(centre: SIMD2<Float>, axis: SIMD2<Float>)] = []
        for f in found where !doors.contains(where: {
            simd_distance($0.centre, f.centre) < 0.8
        }) {
            doors.append((f.centre, f.axis))
        }
        return doors
    }

    /// Places along `band` (column centres), `s` apart: from the band's
    /// place nearest `start`, each next place 0.75-1.3 s on, the
    /// straightest way ahead at the lowest `cost` (metres off where the
    /// loop wants to be), until the band runs out; then again from the
    /// nearest place left, for the band's other pieces. A place within
    /// 0.7 s of `taken` (spots, places already picked) isn't used.
    private static func walk(
        _ band: [SIMD2<Float>], cost: [Float], spacing s: Float,
        from start: SIMD2<Float>?,
        taken: inout [SIMD2<Float>],
        usable: (SIMD2<Float>) -> [Bool]?
    ) -> [(center: SIMD2<Float>, blocked: [Bool])] {
        var out: [(center: SIMD2<Float>, blocked: [Bool])] = []
        let keep = 0.7 * s
        var rejected = Set<Int>()
        func free(_ i: Int) -> Bool {
            !rejected.contains(i)
                && !taken.contains { simd_distance($0, band[i]) < keep }
        }
        /// The first of `order` that is usable; the others tried are
        /// rejected for good.
        func pick(_ order: [Int]) -> (Int, [Bool])? {
            for i in order.prefix(12) {
                if let blocked = usable(band[i]) { return (i, blocked) }
                rejected.insert(i)
            }
            return nil
        }
        var cursor = start
        var chains = 0
        while chains < 50 {
            chains += 1
            let open = band.indices.filter(free)
            guard !open.isEmpty else { break }
            // Nearest the cursor, where the loop wants to be.
            func startScore(_ i: Int) -> Float {
                (cursor.map { simd_distance(band[i], $0) } ?? 0) + 2 * cost[i]
            }
            let first = open.sorted { startScore($0) < startScore($1) }
            guard let (i0, b0) = pick(first) else { continue }
            var cur = band[i0]
            taken.append(cur)
            out.append((cur, b0))
            var dir: SIMD2<Float>?
            while out.count < 400 {
                let next = band.indices.filter { i in
                    let d = simd_distance(band[i], cur)
                    return d >= 0.75 * s && d <= 1.3 * s && free(i)
                }
                guard !next.isEmpty else { break }
                func score(_ i: Int) -> Float {
                    let v = band[i] - cur
                    let d = simd_length(v)
                    let fit = abs(d - s) / s
                    let off = 2 * cost[i]
                    guard let dir else { return -fit - off }
                    return simd_dot(v / d, dir) - 0.3 * fit - off
                }
                guard let (i, b) = pick(
                    next.sorted { score($0) > score($1) })
                else { break }
                let p = band[i]
                dir = simd_normalize(p - cur)
                cur = p
                taken.append(cur)
                out.append((cur, b))
            }
            cursor = cur
        }
        return out
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
