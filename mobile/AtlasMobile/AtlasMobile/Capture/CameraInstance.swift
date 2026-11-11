import CoreMedia
import Foundation
import ImageIO
import INSCameraSDK
import INSCoreMedia

private let kClockSyncTotalSamples = 20
private let kClockSyncBestCount = 3

final class CameraInstance: NSObject, Identifiable {
    let id: String
    let model: String
    let serial: String
    let extrinsic: RigidTransform
    let maskPath: String?

    private(set) var clockOffset: ClockOffset?

    var onDisconnect: (() -> Void)?
    var onStatusUpdate: ((String) -> Void)?

    // MARK: - Thread-safe state

    private let lock = NSLock()
    private var _isConnected = false
    private var _pendingURIs: [(scanIndex: Int, uri: String)] = []
    private var _connectCont: CheckedContinuation<Bool, Never>?
    private var _clockCont: CheckedContinuation<ClockOffset, Never>?
    private var _captureCont: CheckedContinuation<CaptureAck?, Never>?
    private var _captureInFlight = false
    private var _captureGen: UInt64 = 0
    private var _downloadCont: CheckedContinuation<
        Insta360DownloadResult?, Never
    >?
    /// Identifies the download that owns _downloadCont, so a late
    /// reply or timer from an earlier download cannot finish it.
    private var _downloadGen: UInt64 = 0
    /// Failed download attempts per URI; failures are re-queued until
    /// maxDownloadAttempts.
    private var _downloadAttempts: [String: Int] = [:]
    private static let maxDownloadAttempts = 3
    /// systemUptime of the running download's last progress callback.
    private var _downloadProgressAt: Double = 0
    /// A 120 MP X6 photo has 1.7 times the pixels of a 72 MP X5 photo, so
    /// a fixed limit could stop a download that is still receiving data.
    /// A download is given up after this long without progress...
    private static let downloadStallSeconds: Double = 30
    /// ...or after this long in total.
    private static let downloadMaxSeconds: Double = 180
    /// Gets the photo URI from an INSCameraTakePictureResponse push
    /// while a takePicture is in flight. Returns false if that shot has
    /// already ended. New X6 (C9) firmware sends the URI in this push
    /// instead of the takePicture completion; see the SDK sample's
    /// CameraConfigByJsonController.takePictureResponseNotification.
    /// The push carries no request ID, so it goes to the shot in flight.
    private var _photoWaiter: ((String, Double) -> Bool)?
    /// When the in-flight takePicture's completion arrived without a URI.
    private var _captureReplyTs: Double?

    /// Shots captured but not yet downloaded.
    var pendingDownloadCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return _pendingURIs.count
    }

    private(set) var isConnected: Bool {
        get {
            lock.lock()
            defer { lock.unlock() }
            return _isConnected
        }
        set {
            lock.lock()
            defer { lock.unlock() }
            _isConnected = newValue
        }
    }

    private var kvoToken: NSKeyValueObservation?
    private var disconnectToken: NSKeyValueObservation?
    private var photoPushObserver: NSObjectProtocol?
    private var takePictureStateObserver: NSObjectProtocol?

    init(config: CameraConfig) {
        self.id = config.id
        self.model = config.model
        self.serial = config.serial
        self.extrinsic = config.extrinsic
        self.maskPath = config.mask
    }

    deinit {
        stopPhotoPushObserver()
    }

    // MARK: - Connection

    // The heartbeat runs in Insta360ConnectionMonitor for as long as
    // the socket is connected, as in the SDK sample, not per instance.

    func connect() async -> Bool {
        onStatusUpdate?(
            "[\(id)] Waiting for SDK connection..."
        )
        startPhotoPushObserver()

        return await withCheckedContinuation { continuation in
            lock.lock()
            _connectCont = continuation
            lock.unlock()

            func finish(_ result: Bool) {
                self.lock.lock()
                let cont = self._connectCont
                self._connectCont = nil
                self.lock.unlock()
                cont?.resume(returning: result)
            }

            DispatchQueue.main.async { [weak self] in
                guard let self else {
                    finish(false)
                    return
                }

                let socket = INSCameraManager.socket()
                let currentState = socket.cameraState
                self.onStatusUpdate?(
                    "[\(self.id)] state="
                        + "\(Self.stateName(currentState))"
                )

                if currentState == .connected {
                    self.onStatusUpdate?(
                        "[\(self.id)] Already connected!"
                    )
                    self.isConnected = true
                    self.monitorConnection()
                    finish(true)
                    return
                }

                if currentState == .connectFailed {
                    self.onStatusUpdate?(
                        "[\(self.id)] SDK in ConnectFailed state"
                    )
                    finish(false)
                    return
                }

                self.kvoToken = socket.observe(
                    \.cameraState,
                    options: [.new]
                ) { [weak self] _, change in
                    guard let state = change.newValue else {
                        return
                    }
                    DispatchQueue.main.async { [weak self] in
                        guard let self else { return }
                        self.onStatusUpdate?(
                            "[\(self.id)] State -> "
                                + Self.stateName(state)
                        )
                        if state == .connected {
                            self.kvoToken = nil
                            self.isConnected = true
                            self.monitorConnection()
                            self.onStatusUpdate?(
                                "[\(self.id)] Connected!"
                            )
                            finish(true)
                        } else if state == .connectFailed {
                            self.kvoToken = nil
                            finish(false)
                        }
                    }
                }

                DispatchQueue.main.asyncAfter(
                    deadline: .now() + 30
                ) { [weak self] in
                    guard let self else { return }
                    self.kvoToken = nil
                    self.onStatusUpdate?(
                        "[\(self.id)] Timeout waiting"
                            + " for connection"
                    )
                    finish(false)
                }
            }
        }
    }

    private static func stateName(
        _ state: INSCameraState
    ) -> String {
        switch state {
        case .found: return "Found"
        case .synchronized: return "Synchronized"
        case .connected: return "Connected"
        case .connectFailed: return "ConnectFailed"
        case .noConnection: return "NoConnection"
        @unknown default:
            return "unknown(\(state.rawValue))"
        }
    }

    func disconnect() async {
        stopPhotoPushObserver()
        kvoToken = nil
        disconnectToken = nil
        onDisconnect = nil
        onStatusUpdate = nil
        resumeAllContinuations(clearURIs: true)
    }

    func clearPending() {
        lock.lock()
        _pendingURIs.removeAll()
        lock.unlock()
    }

    func clearCaptureState() {
        lock.lock()
        _captureGen += 1
        let cont = _captureCont
        _captureCont = nil
        _photoWaiter = nil
        _captureInFlight = false
        lock.unlock()
        cont?.resume(returning: nil)
    }

    func reconnect(maxRetries: Int = 3) async -> Bool {
        // disconnect() clears the callbacks. Keep them, or this
        // instance logs nothing after Retry.
        let statusUpdate = onStatusUpdate
        let disconnectHandler = onDisconnect
        await disconnect()
        onStatusUpdate = statusUpdate
        onDisconnect = disconnectHandler
        for attempt in 1...maxRetries {
            // Insta360CaptureManager starts the live preview afterwards.
            if await connect() {
                return true
            }
            if attempt < maxRetries {
                try? await Task.sleep(
                    nanoseconds: 2_000_000_000
                )
            }
        }
        return false
    }

    // MARK: - Clock offset

    // syncTimeMsToCamera SETS the camera clock each call.
    // Sample 0 measures the original camera clock (pre-set).
    // Samples 1–N measure how accurately each set landed
    // (residual offset). We select the best by lowest RTT.
    func calibrateClockOffset() async -> ClockOffset {
        let wallSec = Date().timeIntervalSince1970
        let arkitSec = ProcessInfo.processInfo.systemUptime
        let fallback = ClockOffset(
            offsetMs: 0, uncertaintyMs: 999,
            sampleCount: 0, preSetSample: nil,
            samples: [],
            wallClockSec: wallSec,
            arkitTimeSec: arkitSec
        )
        onStatusUpdate?(
            "[\(id)] clockSync: start"
                + " \(kClockSyncTotalSamples) samples"
                + " (main=\(Thread.isMainThread))"
        )
        var preSet: ClockSyncSample?
        var postSetSamples: [ClockSyncSample] = []
        for i in 0..<kClockSyncTotalSamples {
            let sample = await singleSyncSample()
            if let s = sample {
                let tag = i == 0 ? " [pre-set]" : ""
                onStatusUpdate?(String(
                    format: "[\(id)] sync#%d:"
                        + " off=%.1fms rtt=%.0fms%@",
                    i, s.offsetMs, s.roundTripMs, tag
                ))
                if i == 0 {
                    preSet = s
                } else {
                    postSetSamples.append(s)
                }
            } else {
                onStatusUpdate?(
                    "[\(id)] sync#\(i): failed"
                )
            }
        }
        let wallSecEnd = Date().timeIntervalSince1970
        let arkitSecEnd = ProcessInfo.processInfo.systemUptime
        let avgWall = (wallSec + wallSecEnd) / 2
        let avgArkit = (arkitSec + arkitSecEnd) / 2
        guard !postSetSamples.isEmpty else {
            onStatusUpdate?(
                "[\(id)] clockSync: 0 valid samples"
            )
            clockOffset = fallback
            return fallback
        }
        let sorted = postSetSamples.sorted {
            $0.roundTripMs < $1.roundTripMs
        }
        let best = Array(sorted.prefix(kClockSyncBestCount))
        let avgOffset = best.map(\.offsetMs).reduce(
            0, +
        ) / Double(best.count)
        let maxRtt = best.map(\.roundTripMs).max() ?? 0
        let result = ClockOffset(
            offsetMs: avgOffset,
            uncertaintyMs: maxRtt / 2,
            sampleCount: postSetSamples.count,
            preSetSample: preSet,
            samples: postSetSamples,
            wallClockSec: avgWall,
            arkitTimeSec: avgArkit
        )
        onStatusUpdate?(String(
            format: "[\(id)] clockSync: done"
                + " res=%.1fms ±%.0fms (%d samples)",
            result.offsetMs, result.uncertaintyMs,
            postSetSamples.count
        ))
        clockOffset = result
        return result
    }

    private func singleSyncSample()
        async -> ClockSyncSample?
    {
        await withCheckedContinuation { cont in
            let done = NSLock()
            var didResume = false

            func finish(_ result: ClockSyncSample?) {
                done.lock()
                defer { done.unlock() }
                guard !didResume else { return }
                didResume = true
                cont.resume(returning: result)
            }

            DispatchQueue.main.async { [weak self] in
                self?.onStatusUpdate?(
                    "[\(self?.id ?? "?")] syncTime:"
                    + " sending…"
                )
                let sendMs = Date()
                    .timeIntervalSince1970 * 1000
                INSCameraManager.shared()
                    .commandManager
                    .syncTimeMsToCamera(
                        withTryCount: 1,
                        dTimeMsMax: 500
                    ) { dTimeMs, error in
                        let recvMs = Date()
                            .timeIntervalSince1970
                            * 1000
                        if let error {
                            self?.onStatusUpdate?(
                                "[\(self?.id ?? "?")]"
                                + " syncTime err:"
                                + " \(error)"
                            )
                            finish(nil)
                        } else {
                            let midMs = (sendMs + recvMs)
                                / 2
                            let camMs = midMs
                                + Double(dTimeMs)
                            finish(
                                ClockSyncSample(
                                    phoneSendMs: sendMs,
                                    phoneRecvMs: recvMs,
                                    cameraMs: camMs,
                                    offsetMs: Double(
                                        dTimeMs
                                    )
                                )
                            )
                        }
                    }
            }

            DispatchQueue.main.asyncAfter(
                deadline: .now() + 5
            ) { [weak self] in
                done.lock()
                let alreadyDone = didResume
                done.unlock()
                if !alreadyDone {
                    self?.onStatusUpdate?(
                        "[\(self?.id ?? "?")]"
                        + " syncTime: timeout (5s)"
                    )
                }
                finish(nil)
            }
        }
    }

    // MARK: - Capture

    // A shot follows the SDK sample's take-picture page
    // (CameraConfigByJsonController.swift, opened with "Record new"):
    // its live preview runs, and "X5 Take Picture" sends setAppid, then
    // PureShot, then takePicture. The sample sets no photo mode, sensor,
    // FlowState or horizon option and takes no warmup photo, so this
    // doesn't either.

    /// Everything a shot does before takePicture. Calibration calls this
    /// first and then fires with skipPreChecks, so takePicture follows
    /// the ARKit frame closely.
    func prepareForCapture() async -> Bool {
        guard await ensureSDKConnected() else {
            return false
        }
        await prepareShot()
        return true
    }

    func capture(
        arkitTimestamp: Double,
        scanIndex: Int,
        timeout: Double = 45,
        skipPreChecks: Bool = false
    ) async -> Insta360CaptureResult? {
        guard !Task.isCancelled else { return nil }

        lock.lock()
        let alreadyBusy = _captureInFlight
        if !alreadyBusy { _captureInFlight = true }
        lock.unlock()
        if alreadyBusy {
            onStatusUpdate?(
                "[\(id)] capture skipped — in-flight"
            )
            return nil
        }
        defer {
            lock.lock()
            _captureInFlight = false
            lock.unlock()
        }

        if !skipPreChecks {
            guard await ensureSDKConnected() else {
                return nil
            }
            await prepareShot()
        }

        // One takePicture per shot, as the sample sends one per press.
        // The old retry after a missed reply took a second photo.
        let commandTs = ProcessInfo.processInfo.systemUptime
        let captureResult = await attemptTakePicture(
            timeout: timeout
        )

        if let uri = captureResult?.uri {
            lock.lock()
            _pendingURIs.append(
                (scanIndex: scanIndex, uri: uri)
            )
            lock.unlock()
        }

        return Insta360CaptureResult(
            cameraId: id,
            scanIndex: scanIndex,
            arkitTimestamp: arkitTimestamp,
            commandTimestamp: commandTs,
            callbackTimestamp: captureResult?
                .callbackTimestamp ?? commandTs,
            mediaIdentifier: captureResult?.uri ?? ""
        )
    }

    private func waitForConnected(
        maxWait: Double
    ) async -> Bool {
        let start = Date()
        while Date().timeIntervalSince(start) < maxWait {
            let state = await MainActor.run {
                INSCameraManager.socket().cameraState
            }
            if state == .connected {
                onStatusUpdate?(
                    "[\(id)] reconnected after "
                    + "\(Int(Date().timeIntervalSince(start)))s"
                )
                return true
            }
            onStatusUpdate?(
                "[\(id)] waiting… state="
                + Self.stateName(state)
            )
            try? await Task.sleep(for: .seconds(1))
        }
        return false
    }

    private func ensureSDKConnected() async -> Bool {
        let state = await MainActor.run {
            INSCameraManager.socket().cameraState
        }
        if state == .connected { return true }

        onStatusUpdate?(
            "[\(id)] SDK state "
            + Self.stateName(state)
            + " — waiting"
        )
        return await waitForConnected(maxWait: 10)
    }

    /// Brings the camera to where the sample's page is when its button
    /// sends takePicture: the live preview runs, and setAppid and PureShot
    /// have just been sent.
    private func prepareShot() async {
        let previewRunning = await Insta360LivePreview.shared
            .waitUntilRunning(timeout: 10)
        if !previewRunning {
            onStatusUpdate?(
                "[\(id)] live preview not running"
                + " — taking the photo anyway"
            )
        }
        let pureShotSet = await setAppIdAndPureShot()
        if !pureShotSet {
            // The sample's plain "Take Picture" button sends takePicture
            // without them.
            onStatusUpdate?(
                "[\(id)] taking the photo without setAppid/PureShot"
            )
        }
    }

    func waitForCommandReady(
        maxWait: Double = 60
    ) async -> Bool {
        let start = Date()
        while Date().timeIntervalSince(start) < maxWait {
            let ok = await warmUpCommandChannel(
                timeout: 5
            )
            if ok { return true }
            let elapsed = Int(
                Date().timeIntervalSince(start)
            )
            onStatusUpdate?(
                "[\(id)] commands not ready"
                + " (\(elapsed)s) — retrying"
            )
            try? await Task.sleep(for: .seconds(1))
        }
        return false
    }

    func warmUpCommandChannel(
        timeout: Double = 15
    ) async -> Bool {
        await withCheckedContinuation {
            (cont: CheckedContinuation<Bool, Never>)
                in
            let done = NSLock()
            var didResume = false

            func finish(_ result: Bool) {
                done.lock()
                defer { done.unlock() }
                guard !didResume else { return }
                didResume = true
                cont.resume(returning: result)
            }

            DispatchQueue.main.async { [weak self] in
                let types: [NSNumber] = [
                    NSNumber(
                        value: INSCameraOptionsType
                            .storageState.rawValue
                    ),
                ]
                INSCameraManager.shared()
                    .commandManager
                    .getOptionsWithTypes(
                        types
                    ) { error, _, _ in
                        if let error {
                            self?.onStatusUpdate?(
                                "[\(self?.id ?? "?")]"
                                + " command check: "
                                + "\(error)"
                            )
                        } else {
                            self?.onStatusUpdate?(
                                "[\(self?.id ?? "?")]"
                                + " command check OK"
                            )
                        }
                        finish(error == nil)
                    }
            }

            DispatchQueue.main.asyncAfter(
                deadline: .now() + timeout
            ) { [weak self] in
                done.lock()
                let alreadyDone = didResume
                done.unlock()
                if !alreadyDone {
                    self?.onStatusUpdate?(
                        "[\(self?.id ?? "?")]"
                        + " command check timeout"
                        + " (\(Int(timeout))s)"
                    )
                }
                finish(false)
            }
        }
    }

    /// Sends what the SDK sample's "X5 Take Picture" sends before every
    /// takePicture (CameraConfigByJsonController.swift): setAppid, then
    /// raw capture type PureShot (3) for function mode 8 (HDR_IMAGE),
    /// option type 25. The sample's comment: the camera accepts photo
    /// settings only after setAppid, and without it the photo still
    /// includes a DNG. Returns whether both succeeded.
    private func setAppIdAndPureShot() async -> Bool {
        await withCheckedContinuation {
            (continuation: CheckedContinuation<Bool, Never>)
                in
            let resumed = NSLock()
            var didResume = false

            func finish(_ ok: Bool) {
                resumed.lock()
                defer { resumed.unlock() }
                guard !didResume else { return }
                didResume = true
                continuation.resume(returning: ok)
            }

            DispatchQueue.main.async { [weak self] in
                let commands = INSCameraManager.shared()
                    .commandManager
                commands.setAppidCompletion { [weak self] error in
                    if let error {
                        self?.onStatusUpdate?(
                            "[\(self?.id ?? "?")] "
                            + "setAppid err: \(error)"
                        )
                        finish(false)
                        return
                    }
                    let photo = INSPhotographyOptions()
                    photo.rawCaptureType = UInt8(
                        INSCameraRawCaptureType.pureshot.rawValue
                    )
                    let types: [NSNumber] = [
                        NSNumber(
                            value: INSPhotographyOptionsType
                                .rawCaptureType.rawValue
                        ),
                    ]
                    commands.setPhotographyOptions(
                        photo,
                        for: INSCameraFunctionMode(functionMode: 8),
                        types: types
                    ) { [weak self] error, _ in
                        if let error {
                            self?.onStatusUpdate?(
                                "[\(self?.id ?? "?")] "
                                + "PureShot err: \(error)"
                            )
                            finish(false)
                        } else {
                            self?.onStatusUpdate?(
                                "[\(self?.id ?? "?")] "
                                + "setAppid + PureShot set"
                            )
                            finish(true)
                        }
                    }
                }
            }

            DispatchQueue.main.asyncAfter(
                deadline: .now() + 15
            ) { [weak self] in
                resumed.lock()
                let alreadyDone = didResume
                resumed.unlock()
                if !alreadyDone {
                    self?.onStatusUpdate?(
                        "[\(self?.id ?? "?")] "
                        + "setAppid/PureShot timeout"
                    )
                }
                finish(false)
            }
        }
    }

    private typealias CaptureAck = (
        uri: String, callbackTimestamp: Double
    )

    /// The options the sample's "X5 Take Picture" sets: raw capture type
    /// PureShot and countDown 0. Like the sample, it leaves mode at its
    /// default (normal) and currenExposureOptions unset.
    private static func takePictureOptions() -> INSTakePictureOptions {
        let opts = INSTakePictureOptions()
        opts.rawCaptureType = .pureshot
        opts.countDown = 0
        return opts
    }

    /// Sends one takePicture and waits for its photo URI. Old firmware
    /// returns the URI in the takePicture completion. New X6 (C9)
    /// firmware returns it in an INSCameraTakePictureResponse push
    /// instead (handlePhotoPush). The SDK sample handles both.
    private func attemptTakePicture(
        timeout: Double
    ) async -> CaptureAck? {
        return await withTaskCancellationHandler {
            await withCheckedContinuation {
                continuation in
                self.lock.lock()
                self._captureGen += 1
                let gen = self._captureGen
                self._captureCont = continuation
                self._captureReplyTs = nil
                self.lock.unlock()

                // Ends this attempt once. Returns false if it had
                // already ended (timeout, cancel, clearCaptureState).
                func finish(_ ack: CaptureAck?) -> Bool {
                    self.lock.lock()
                    guard gen == self._captureGen,
                          let cont = self._captureCont
                    else {
                        self.lock.unlock()
                        return false
                    }
                    self._captureCont = nil
                    self._photoWaiter = nil
                    self.lock.unlock()
                    cont.resume(returning: ack)
                    return true
                }

                // The push's time stands in for the reply time only if
                // the completion hasn't arrived yet.
                self.lock.lock()
                self._photoWaiter = { uri, pushTs in
                    self.lock.lock()
                    let replyTs = self._captureReplyTs
                    self.lock.unlock()
                    return finish(
                        CaptureAck(
                            uri: uri,
                            callbackTimestamp: replyTs ?? pushTs
                        )
                    )
                }
                self.lock.unlock()

                DispatchQueue.main.async {
                    INSCameraManager.shared()
                        .commandManager
                        .takePicture(
                            with: Self.takePictureOptions()
                        ) { error, photoInfo in
                            let cbTs = ProcessInfo
                                .processInfo.systemUptime
                            let uri = Self.photoURI(photoInfo)
                            let hdr = photoInfo?.hdrUris?.count ?? 0
                            let burst = photoInfo?.burstUris?.count
                                ?? 0
                            var note = "[\(self.id)] cb:"
                                + " gen=\(gen)"
                                + " uri=\(uri ?? "nil")"
                            if hdr > 0 { note += " hdr=\(hdr)" }
                            if burst > 0 { note += " burst=\(burst)" }
                            if let error {
                                note += " err=\(error)"
                            } else if uri == nil {
                                note += " (waiting for photo push)"
                            }
                            self.onStatusUpdate?(note)
                            if error != nil {
                                _ = finish(nil)
                            } else if let uri {
                                _ = finish(
                                    CaptureAck(
                                        uri: uri,
                                        callbackTimestamp: cbTs
                                    )
                                )
                            } else {
                                // No URI and no error: the URI comes in
                                // a push. Keep waiting until the timeout.
                                self.lock.lock()
                                if gen == self._captureGen {
                                    self._captureReplyTs = cbTs
                                }
                                self.lock.unlock()
                            }
                        }
                }

                DispatchQueue.main.asyncAfter(
                    deadline: .now() + timeout
                ) {
                    if finish(nil) {
                        self.onStatusUpdate?(
                            "[\(self.id)] takePicture"
                            + " timeout"
                            + " (\(Int(timeout))s)"
                        )
                    }
                }
            }
        } onCancel: {
            self.lock.lock()
            self._captureGen += 1
            let cont = self._captureCont
            self._captureCont = nil
            self._photoWaiter = nil
            self.lock.unlock()
            cont?.resume(returning: nil)
        }
    }

    /// Photo URI from a takePicture reply. A single shot fills `uri`;
    /// HDR or AEB shots fill `hdrUris` and bursts fill `burstUris`, which
    /// the SDK sample also reads. The SDK documents HDR order as
    /// [EV0, -EV, +EV], so the first HDR URI is the normal exposure.
    private static func photoURI(
        _ info: INSCameraPhotoInfo?
    ) -> String? {
        if let u = info?.uri, !u.isEmpty { return u }
        if let u = info?.hdrUris?.first(where: { !$0.isEmpty }) {
            return u
        }
        return info?.burstUris?.first(where: { !$0.isEmpty })
    }

    // MARK: - Photo URI push

    /// Observes on NotificationCenter.default, as the SDK sample does.
    /// Also logs INSCameraTakePictureStateUpdate, which the sample's root
    /// page observes, to show how far a shot got if no URI arrives.
    private func startPhotoPushObserver() {
        if photoPushObserver == nil {
            photoPushObserver = NotificationCenter.default.addObserver(
                forName: .INSCameraTakePictureResponse,
                object: nil,
                queue: nil
            ) { [weak self] note in
                self?.handlePhotoPush(note)
            }
        }
        if takePictureStateObserver == nil {
            takePictureStateObserver = NotificationCenter.default
                .addObserver(
                    forName: .INSCameraTakePictureStateUpdate,
                    object: nil,
                    queue: nil
                ) { [weak self] note in
                    let state = note.userInfo?["state"]
                        .map { "\($0)" } ?? "?"
                    self?.onStatusUpdate?(
                        "[\(self?.id ?? "?")] take picture state: \(state)"
                    )
                }
        }
    }

    private func stopPhotoPushObserver() {
        for token in [photoPushObserver, takePictureStateObserver] {
            if let token {
                NotificationCenter.default.removeObserver(token)
            }
        }
        photoPushObserver = nil
        takePictureStateObserver = nil
    }

    /// userInfo["photo"] is an INSCameraPhotoInfo. The push goes to the
    /// takePicture in flight; with none in flight it is ignored (for
    /// example a photo taken with the camera's own button).
    private func handlePhotoPush(_ note: Notification) {
        let info = note.userInfo?["photo"] as? INSCameraPhotoInfo
        guard let uri = Self.photoURI(info) else {
            onStatusUpdate?("[\(id)] photo push without a URI")
            return
        }
        lock.lock()
        let waiter = _photoWaiter
        lock.unlock()
        let used = waiter?(
            uri, ProcessInfo.processInfo.systemUptime
        ) ?? false
        onStatusUpdate?(
            "[\(id)] photo push: uri=\(uri)"
            + (used ? "" : " (no shot waiting, ignored)")
        )
    }

    // MARK: - Download

    func downloadPending(
        into directory: URL
    ) async -> [Insta360DownloadResult] {
        // Take the queue only while connected; otherwise leave it for a
        // later pass instead of dropping the shots.
        lock.lock()
        let connected = _isConnected
        let toDownload = connected ? _pendingURIs : []
        if connected { _pendingURIs.removeAll() }
        lock.unlock()

        guard !toDownload.isEmpty else {
            return []
        }

        var results: [Insta360DownloadResult] = []
        var failed: [(scanIndex: Int, uri: String)] = []
        for item in toDownload {
            if let r = await downloadOne(
                scanIndex: item.scanIndex,
                uri: item.uri,
                into: directory
            ) {
                results.append(r)
            } else {
                failed.append(item)
            }
        }

        // Put failures back so the next pass (the next shot, or the
        // retry at session end) tries them again.
        if !failed.isEmpty {
            var requeued = 0
            lock.lock()
            for item in failed {
                let n = (_downloadAttempts[item.uri] ?? 0) + 1
                _downloadAttempts[item.uri] = n
                if n < Self.maxDownloadAttempts {
                    _pendingURIs.append(item)
                    requeued += 1
                }
            }
            lock.unlock()
            onStatusUpdate?(
                "[\(id)] \(failed.count) download(s) failed,"
                + " \(requeued) re-queued"
            )
        }
        return results
    }

    // MARK: - Private

    private func resumeAllContinuations(
        clearURIs: Bool
    ) {
        let fallback = ClockOffset(
            offsetMs: 0, uncertaintyMs: 999,
            sampleCount: 0, preSetSample: nil,
            samples: [],
            wallClockSec: Date().timeIntervalSince1970,
            arkitTimeSec: ProcessInfo.processInfo
                .systemUptime
        )

        lock.lock()
        _isConnected = false
        _captureInFlight = false
        _photoWaiter = nil
        _captureReplyTs = nil
        if clearURIs {
            _pendingURIs.removeAll()
            _downloadAttempts.removeAll()
        }
        let connectCont = _connectCont
        let clockCont = _clockCont
        let captureCont = _captureCont
        let downloadCont = _downloadCont
        _connectCont = nil
        _clockCont = nil
        _captureCont = nil
        _downloadCont = nil
        lock.unlock()

        connectCont?.resume(returning: false)
        clockCont?.resume(returning: fallback)
        captureCont?.resume(returning: nil)
        downloadCont?.resume(returning: nil)
    }

    private func downloadOne(
        scanIndex: Int,
        uri: String,
        into directory: URL
    ) async -> Insta360DownloadResult? {
        let destURL = directory
            .appendingPathComponent(id)
            .appendingPathComponent(
                String(
                    format: "scan_%03d.jpg",
                    scanIndex
                )
            )

        try? FileManager.default.createDirectory(
            at: destURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try? FileManager.default.removeItem(at: destURL)

        let cameraId = self.id
        let started = ProcessInfo.processInfo.systemUptime
        return await withCheckedContinuation {
            continuation in
            lock.lock()
            _downloadGen += 1
            let gen = _downloadGen
            _downloadCont = continuation
            _downloadProgressAt = started
            lock.unlock()

            // Only this download may finish its continuation. Without
            // the check, this download's timeout (or a late reply)
            // could finish the next download with nil. Returns whether
            // this call finished it, so a check that runs after the
            // download ended is not logged as a timeout.
            func finish(
                _ value: Insta360DownloadResult?
            ) -> Bool {
                self.lock.lock()
                guard gen == self._downloadGen,
                      let cont = self._downloadCont
                else {
                    self.lock.unlock()
                    return false
                }
                self._downloadCont = nil
                self.lock.unlock()
                cont.resume(returning: value)
                return true
            }

            // The SDK sample downloads every file through the command
            // manager and never uses INSCameraHTTPManager.
            DispatchQueue.main.async {
                INSCameraManager.shared().commandManager.fetchResource(
                    withURI: uri,
                    toLocalFile: destURL,
                    progress: { _ in
                        let now = ProcessInfo.processInfo
                            .systemUptime
                        self.lock.lock()
                        if gen == self._downloadGen {
                            self._downloadProgressAt = now
                        }
                        self.lock.unlock()
                    }
                ) { error in
                    DispatchQueue.main.async {
                        if let error {
                            if finish(nil) {
                                self.onStatusUpdate?(
                                    "[\(cameraId)] download failed:"
                                    + " \(uri): \(error)"
                                )
                            }
                            return
                        }
                        let exifTs = Self
                            .extractExifTimestamp(
                                from: destURL
                            )
                        let done = finish(Insta360DownloadResult(
                            cameraId: cameraId,
                            scanIndex: scanIndex,
                            localURL: destURL,
                            mediaType: .equirectangular,
                            exifTimestamp: exifTs
                        ))
                        if done {
                            let bytes = (try? FileManager.default
                                .attributesOfItem(
                                    atPath: destURL.path
                                )[.size] as? Int) ?? 0
                            let detail = String(
                                format: "%.1f MB, %.1f s",
                                Double(bytes) / 1_048_576,
                                ProcessInfo.processInfo
                                    .systemUptime - started)
                            self.onStatusUpdate?(
                                "[\(cameraId)] downloaded"
                                + " \(destURL.lastPathComponent)"
                                + " (\(detail))"
                            )
                        }
                    }
                }
            }

            // Checks every 5 s; gives up after downloadStallSeconds
            // without progress or downloadMaxSeconds in total.
            func watchDownload() {
                DispatchQueue.main.asyncAfter(
                    deadline: .now() + 5
                ) {
                    let now = ProcessInfo.processInfo.systemUptime
                    self.lock.lock()
                    let pending = gen == self._downloadGen
                        && self._downloadCont != nil
                    let idle = now - self._downloadProgressAt
                    self.lock.unlock()
                    guard pending else { return }
                    let total = now - started
                    if idle < Self.downloadStallSeconds,
                       total < Self.downloadMaxSeconds
                    {
                        watchDownload()
                        return
                    }
                    if finish(nil) {
                        self.onStatusUpdate?(
                            "[\(cameraId)] download timeout"
                            + String(
                                format: " (%.0f s, %.0f s without data)",
                                total, idle)
                            + ": \(uri)"
                        )
                    }
                }
            }
            watchDownload()
        }
    }

    private static let disconnectStates:
        Set<INSCameraState> = [
            .connectFailed, .noConnection,
        ]

    private func monitorConnection() {
        disconnectToken = INSCameraManager.socket()
            .observe(
                \.cameraState, options: [.new]
            ) { [weak self] _, change in
                guard let state = change.newValue else {
                    return
                }
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.isConnected,
                          Self.disconnectStates
                              .contains(state)
                    else { return }
                    print(
                        "[Insta360] disconnect detected:"
                            + " \(Self.stateName(state))"
                    )
                    self.disconnectToken = nil
                    self.resumeAllContinuations(
                        clearURIs: false
                    )
                    self.onDisconnect?()
                }
            }
    }

    // MARK: - EXIF

    private static let exifDateFormatter: DateFormatter = {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy:MM:dd HH:mm:ss"
        fmt.locale = Locale(identifier: "en_US_POSIX")
        fmt.timeZone = .current
        return fmt
    }()

    static func extractExifTimestamp(
        from url: URL
    ) -> Double? {
        guard let source = CGImageSourceCreateWithURL(
            url as CFURL, nil
        ) else { return nil }
        guard let props = CGImageSourceCopyPropertiesAtIndex(
            source, 0, nil
        ) as? [String: Any] else { return nil }
        guard let exif = props[
            kCGImagePropertyExifDictionary as String
        ] as? [String: Any] else { return nil }
        guard let dateStr = exif[
            kCGImagePropertyExifDateTimeOriginal as String
        ] as? String else { return nil }
        guard let base = exifDateFormatter.date(
            from: dateStr
        ) else { return nil }
        var ts = base.timeIntervalSince1970
        if let sub = exif[
            kCGImagePropertyExifSubsecTimeOriginal as String
        ] as? String, let frac = Double("0." + sub) {
            ts += frac
        }
        return ts
    }
}
