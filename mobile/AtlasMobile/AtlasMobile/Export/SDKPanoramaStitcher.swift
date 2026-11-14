import CoreGraphics
import Foundation
import ImageIO
import INSCoreMedia
import simd

/// The "Insta360 SDK stitcher" 360 lens model (PanoramaLensModel).
///
/// The Insta360 SDK stitches each 360 photo into an equirectangular
/// panorama (INSExportImageSimplify, the SDK's photo export, as in the SDK
/// sample's ExportVideoViewController), using the lens calibration the
/// camera stores in the photo. The SDK's panorama has its own orientation,
/// so it is matched against the app's own panorama of the same photo
/// (DualFisheyeStitcher with the curve) and turned into the app's 360
/// frame: the frame the rig extrinsic, the export's tiles and rig
/// calibration use. A photo whose panorama fails or doesn't match keeps
/// its raw photo, which the export cuts with the curve.
///
/// Panoramas go to <session>/sdk_panoramas/<camera>/scan_NNN.jpg with a
/// scan_NNN.json record, and later exports reuse them, failures included;
/// delete the folder to redo them. They live outside raw/, whose folders
/// buildExportData reads as cameras.
enum SDKPanoramaStitcher {
    static let folderName = "sdk_panoramas"
    /// Panorama width; the height is half. The export's own panorama of
    /// a dual-fisheye photo is at most 8192 wide too.
    static let outputWidth = 8192
    /// Width the SDK and app panoramas are compared at.
    private static let matchWidth = 1024
    /// A panorama is used only if its best right-angle orientation
    /// matches the app's panorama at least this well (normalised
    /// cross-correlation of detail at 256 wide, 1 = identical)...
    static let minCoarse: Float = 0.2
    /// ...and beats the next best by this much, so the orientation isn't
    /// a guess. The fine-detail score isn't a test: it drops wherever
    /// the SDK's lens differs from the app's curve, which is the point
    /// of the SDK. In a simulation on four session 21-53-40 photos
    /// (white wall and floor included), right orientations scored
    /// 0.30-0.71 and led by 0.27 or more, the turn found to within 0.31°
    /// even with the SDK lens 2° off the curve at 50°. Wrong orientations
    /// scored 0.05 or less; another photo's panorama scored 0.07 at best.
    static let minMargin: Float = 0.1
    /// Bump when panoramas or records change, so old ones are redone.
    private static let recordVersion = 1

    struct Record: Codable {
        var version: Int
        var ok: Bool
        /// Best right-angle orientation's score, its lead over the next
        /// one, and the fine-detail score after refining.
        var coarse: Float?
        var margin: Float?
        var match: Float?
        /// Row-major rotation taking app-frame directions to the SDK
        /// panorama's frame.
        var rotation: [Float]?
        var seconds: Double?
        var note: String
    }

    // MARK: - Session

    /// Replaces each 360 photo in `scans` with its SDK panorama, making
    /// the ones that don't exist yet. A photo it can't use keeps its raw
    /// photo. `onProgress(done, total)` is called after each photo, on
    /// the calling thread.
    static func prepare(
        scans: [ScanExportData],
        config: MultiCameraConfig,
        sessionDirectory: URL,
        onProgress: ((Int, Int) -> Void)? = nil
    ) -> (scans: [ScanExportData], note: String) {
        let total = scans.reduce(0) { $0 + $1.insta360ImageURLs.count }
        guard total > 0 else {
            return (scans, "SDK stitcher: no 360 photos")
        }
        let start = Date()
        var done = 0, made = 0, reused = 0
        var kept: [String] = []
        var rotations: [simd_float3x3] = []
        var out: [ScanExportData] = []
        for scan in scans {
            guard !scan.insta360ImageURLs.isEmpty else {
                out.append(scan)
                continue
            }
            var urls = scan.insta360ImageURLs
            for camId in scan.insta360ImageURLs.keys.sorted() {
                guard let raw = scan.insta360ImageURLs[camId] else {
                    continue
                }
                let model = config.cameras.first { $0.id == camId }?
                    .model ?? ""
                let result = autoreleasepool {
                    panorama(raw: raw, scanName: scan.scanName,
                             cameraId: camId, model: model,
                             sessionDirectory: sessionDirectory)
                }
                done += 1
                let what: String
                if let url = result.url {
                    urls[camId] = url
                    if result.fresh { made += 1 } else { reused += 1 }
                    if let r = result.record.rotation, r.count == 9 {
                        rotations.append(matrix(rows: r))
                    }
                    what = result.fresh ? "made" : "reused"
                } else {
                    kept.append("\(scan.scanName) (\(result.record.note))")
                    what = "kept the raw photo"
                }
                ExportTrace.note("SDK stitcher \(done)/\(total):"
                    + " \(scan.scanName) \(what): \(result.record.note)")
                onProgress?(done, total)
            }
            out.append(ScanExportData(
                scanName: scan.scanName,
                arkitPose: scan.arkitPose,
                intrinsics: scan.intrinsics,
                imageWidth: scan.imageWidth,
                imageHeight: scan.imageHeight,
                depthBinURL: scan.depthBinURL,
                smoothedDepthBinURL: scan.smoothedDepthBinURL,
                iphoneImageURL: scan.iphoneImageURL,
                insta360ImageURLs: urls,
                insta360CommandTimestamp: scan.insta360CommandTimestamp))
        }
        var note = "SDK stitcher: \(made + reused) of \(total) 360 photos"
            + " as SDK panoramas (\(made) made, \(reused) reused,"
            + " \(Int(Date().timeIntervalSince(start))) s)"
        // The SDK's frame is fixed to the camera, so every photo should
        // need the same turn; a spread means the SDK levelled photos.
        if let first = rotations.first, rotations.count > 1 {
            let spread = rotations
                .map { angleDeg(first.transpose * $0) }.max() ?? 0
            note += String(
                format: "; their turns to the app's frame agree within %.2f°",
                spread)
        }
        if !kept.isEmpty {
            note += "; kept the raw photo, cut with the curve, for"
                + " \(kept.count): " + kept.joined(separator: ", ")
        }
        return (out, note)
    }

    // MARK: - One photo

    /// The SDK panorama of one photo: its URL (nil if the raw photo is
    /// kept), its record, and whether it was made now.
    private static func panorama(
        raw: URL, scanName: String, cameraId: String, model: String,
        sessionDirectory: URL
    ) -> (url: URL?, record: Record, fresh: Bool) {
        let fm = FileManager.default
        let dir = sessionDirectory
            .appendingPathComponent(folderName)
            .appendingPathComponent(cameraId)
        let outURL = dir.appendingPathComponent("\(scanName).jpg")
        let recURL = dir.appendingPathComponent("\(scanName).json")
        if let data = try? Data(contentsOf: recURL),
           let rec = try? JSONDecoder().decode(Record.self, from: data),
           rec.version == recordVersion,
           !rec.ok || fm.fileExists(atPath: outURL.path)
        {
            return (rec.ok ? outURL : nil, rec, false)
        }
        try? fm.createDirectory(
            at: dir, withIntermediateDirectories: true)
        let started = Date()
        var rec = make(raw: raw, model: model, output: outURL)
        rec.seconds = Date().timeIntervalSince(started)
        if !rec.ok { try? fm.removeItem(at: outURL) }
        if let data = try? JSONEncoder().encode(rec) {
            try? data.write(to: recURL)
        }
        return (rec.ok ? outURL : nil, rec, true)
    }

    private static func make(
        raw: URL, model: String, output: URL
    ) -> Record {
        var rec = Record(version: recordVersion, ok: false, note: "")
        let fm = FileManager.default
        let sdkOut = fm.temporaryDirectory.appendingPathComponent(
            "sdk_panorama_\(UUID().uuidString).jpg")
        defer { try? fm.removeItem(at: sdkOut) }

        // 1. The SDK's panorama.
        if let why = stitchWithSDK(raw: raw, output: sdkOut) {
            rec.note = why
            return rec
        }

        // 2. Its orientation, matched against the app's panorama.
        let w = matchWidth, h = matchWidth / 2
        let roll = DualFisheyeStitcher.rollCorrection(forModel: model)
        let lens = DualFisheyeStitcher.lensCurve(forModel: model)
        guard let ref = DualFisheyeStitcher.stitchToPixels(
                url: raw, outputWidth: w, rollDeg: roll,
                allowSubsample: true, lens: lens),
              ref.w == w, ref.h == h,
              let sdkGray = loadGray(url: sdkOut, width: w, height: h)
        else {
            rec.note = "could not load the photos to match"
            return rec
        }
        let fit = align(
            reference: grayFromRGBA(ref.px, count: w * h),
            panorama: sdkGray, width: w, height: h)
        rec.coarse = fit.coarse
        rec.margin = fit.margin
        rec.match = fit.score
        rec.rotation = rows(fit.rotation)
        guard fit.coarse >= minCoarse, fit.margin >= minMargin else {
            rec.note = String(
                format: "couldn't tell how the SDK panorama is turned:"
                    + " best orientation %.2f, next %.2f",
                fit.coarse, fit.coarse - fit.margin)
            return rec
        }

        // 3. The panorama turned into the app's frame.
        guard writeInAppFrame(
            sdkPanorama: sdkOut, rotation: fit.rotation, to: output)
        else {
            rec.note = "could not write the panorama"
            return rec
        }
        rec.ok = true
        rec.note = String(
            format: "turned %.1f° to the app's frame (orientation %.2f,"
                + " next %.2f; fine detail %.2f)",
            angleDeg(fit.rotation), fit.coarse,
            fit.coarse - fit.margin, fit.score)
        return rec
    }

    /// Runs the SDK's photo export; nil on success, else why not. Photos
    /// are saved with .jpg; if the SDK doesn't take that, it gets the
    /// photo again with the camera's .insp extension.
    private static func stitchWithSDK(raw: URL, output: URL) -> String? {
        let fm = FileManager.default
        func run(_ input: URL) -> String? {
            try? fm.removeItem(at: output)
            let exporter = INSExportImageSimplify()
            exporter.width = Int32(outputWidth)
            exporter.height = Int32(outputWidth / 2)
            // The API doc says it returns nil on success. Whatever
            // comes back, the written file decides.
            let reply: Error? = exporter.exportImage(
                withInputUrl: input, outputUrl: output)
            if isPanoramaFile(output) { return nil }
            return reply.map { "SDK: \($0.localizedDescription)" }
                ?? "the SDK wrote no panorama"
        }
        guard let first = run(raw) else { return nil }
        let insp = fm.temporaryDirectory.appendingPathComponent(
            "sdk_photo_\(UUID().uuidString).insp")
        defer { try? fm.removeItem(at: insp) }
        if (try? fm.linkItem(at: raw, to: insp)) == nil,
           (try? fm.copyItem(at: raw, to: insp)) == nil
        {
            return first
        }
        return run(insp).map { "\(first); as .insp: \($0)" }
    }

    private static func isPanoramaFile(_ url: URL) -> Bool {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let props = CGImageSourceCopyPropertiesAtIndex(
                source, 0, nil) as? [CFString: Any],
              let w = props[kCGImagePropertyPixelWidth] as? Int,
              let h = props[kCGImagePropertyPixelHeight] as? Int,
              w >= matchWidth, h > 0
        else { return false }
        let ratio = Double(w) / Double(h)
        return ratio > 1.95 && ratio < 2.05
    }

    // MARK: - Writing the panorama in the app's frame

    /// Resamples the SDK panorama so app-frame direction d shows what the
    /// SDK panorama has at rotation · d, in stitch's layout, and writes
    /// it as JPEG marked with DualFisheyeStitcher.panoramaDescription.
    private static func writeInAppFrame(
        sdkPanorama url: URL, rotation M: simd_float3x3, to output: URL
    ) -> Bool {
        // At most outputWidth wide, in case the SDK writes a larger
        // panorama than asked: two RGBA copies of it are held below.
        let input: (px: [UInt8], w: Int, h: Int)? = autoreleasepool {
            guard let source = CGImageSourceCreateWithURL(
                    url as CFURL, nil),
                  let cg = CGImageSourceCreateThumbnailAtIndex(
                    source, 0, [
                        kCGImageSourceThumbnailMaxPixelSize: outputWidth,
                        kCGImageSourceCreateThumbnailFromImageAlways: true,
                    ] as CFDictionary)
            else { return nil }
            let w = cg.width, h = cg.height
            var px = [UInt8](repeating: 0, count: w * h * 4)
            let drew = px.withUnsafeMutableBytes { buf -> Bool in
                guard let ctx = CGContext(
                    data: buf.baseAddress, width: w, height: h,
                    bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
                else { return false }
                ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
                return true
            }
            return drew ? (px, w, h) : nil
        }
        guard let input, input.w > 1, input.h > 1 else { return false }
        let w = input.w, h = input.h
        let W = Float(w), H = Float(h)
        let sinLon = (0..<w).map { sin(2 * Float.pi * (Float($0) / W - 0.5)) }
        let cosLon = (0..<w).map { cos(2 * Float.pi * (Float($0) / W - 0.5)) }

        var outPx = [UInt8](repeating: 0, count: w * h * 4)
        input.px.withUnsafeBufferPointer { src in
            outPx.withUnsafeMutableBufferPointer { dst in
                guard let base = dst.baseAddress else { return }
                DispatchQueue.concurrentPerform(iterations: h) { v in
                    let lat = Float.pi * (0.5 - Float(v) / H)
                    let cl = cos(lat), sl = sin(lat)
                    for u in 0..<w {
                        let d = M * SIMD3<Float>(
                            cl * sinLon[u], -sl, cl * cosLon[u])
                        // ImageIO-decoded panoramas from the SDK put
                        // pixel centres at +0.5.
                        let (r, g, b) = DualFisheyeStitcher.sampleERP(
                            src, width: w, height: h, direction: d,
                            centreOffset: 0.5)
                        let o = (v * w + u) * 4
                        base[o] = UInt8(clamping: Int(r + 0.5))
                        base[o + 1] = UInt8(clamping: Int(g + 0.5))
                        base[o + 2] = UInt8(clamping: Int(b + 0.5))
                        base[o + 3] = 255
                    }
                }
            }
        }

        return outPx.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(
                    data: buf.baseAddress, width: w, height: h,
                    bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue),
                  let image = ctx.makeImage(),
                  let dest = CGImageDestinationCreateWithURL(
                    output as CFURL, "public.jpeg" as CFString, 1, nil)
            else { return false }
            let tiff: [CFString: Any] = [
                kCGImagePropertyTIFFImageDescription:
                    DualFisheyeStitcher.panoramaDescription,
            ]
            let props: [CFString: Any] = [
                kCGImageDestinationLossyCompressionQuality: 0.92,
                kCGImagePropertyTIFFDictionary: tiff,
            ]
            CGImageDestinationAddImage(dest, image, props as CFDictionary)
            return CGImageDestinationFinalize(dest)
        }
    }

    // MARK: - Matching the SDK panorama to the app's frame

    private struct Level {
        let w: Int
        let h: Int
        /// The SDK panorama's fine detail at this size.
        let pano: [Float]
        /// App-frame directions of the reference pixels compared...
        let dirs: [SIMD3<Float>]
        /// ...their fine detail in the app's panorama...
        let ref: [Float]
        /// ...and their weights.
        let weight: [Float]
    }

    /// The rotation taking app-frame directions to the SDK panorama's
    /// frame. The best of the 24 right-angle turns on a 256-wide copy,
    /// then refined in smaller and smaller turns up to 1024 wide. Scores:
    /// normalised cross-correlation of detail (image minus a 7-pixel box
    /// blur), weighted by area. coarse: the best right-angle turn's
    /// score; margin: its lead over the next; score: after refining.
    private static func align(
        reference: [Float], panorama: [Float], width: Int, height: Int
    ) -> (rotation: simd_float3x3, score: Float, coarse: Float,
          margin: Float) {
        var refs = [reference], panos = [panorama]
        var sizes = [(w: width, h: height)]
        for _ in 0..<2 {
            let s = sizes[sizes.count - 1]
            refs.append(half(refs[refs.count - 1], w: s.w, h: s.h))
            panos.append(half(panos[panos.count - 1], w: s.w, h: s.h))
            sizes.append((w: s.w / 2, h: s.h / 2))
        }
        // Where reference pixel u sits, in pixels: stitch puts pixel u at
        // u; each halving averages two pixels, adding 1/4 then 3/8.
        let refOffsets: [Float] = [0, 0.25, 0.375]
        let strides = [2, 1, 1]
        var levels: [Level] = []
        for l in 0..<3 {
            let w = sizes[l].w, h = sizes[l].h
            let refDetail = detail(refs[l], w: w, h: h)
            let W = Float(w), H = Float(h), o = refOffsets[l]
            var dirs: [SIMD3<Float>] = []
            var vals: [Float] = []
            var wts: [Float] = []
            for v in Swift.stride(from: 0, to: h, by: strides[l]) {
                let lat = Float.pi * (0.5 - (Float(v) + o) / H)
                let cl = cos(lat), sl = sin(lat)
                for u in Swift.stride(from: 0, to: w, by: strides[l]) {
                    let lon = 2 * Float.pi * ((Float(u) + o) / W - 0.5)
                    let d = SIMD3<Float>(cl * sin(lon), -sl, cl * cos(lon))
                    // Over 60° from both lens axes the measured curve is
                    // only extended, so the app's panorama counts less.
                    let seam: Float = abs(d.z) < 0.5 ? 0.2 : 1
                    dirs.append(d)
                    vals.append(refDetail[v * w + u])
                    wts.append(cl * seam)
                }
            }
            levels.append(Level(
                w: w, h: h, pano: detail(panos[l], w: w, h: h),
                dirs: dirs, ref: vals, weight: wts))
        }

        let coarse = rightAngleRotations()
            .map { (rotation: $0, score: ncc($0, levels[2])) }
            .sorted { $0.score > $1.score }
        var best = coarse[0].rotation
        let margin = coarse[0].score - coarse[1].score
        var score = coarse[0].score
        let schedule: [(level: Int, stepsDeg: [Float])] = [
            (2, [8, 4]), (1, [2, 1]), (0, [0.5, 0.25, 0.12, 0.06]),
        ]
        for stage in schedule {
            let level = levels[stage.level]
            score = ncc(best, level)
            for stepDeg in stage.stepsDeg {
                var improved = true
                var rounds = 0
                while improved && rounds < 12 {
                    improved = false
                    rounds += 1
                    for axis in 0..<3 {
                        for sign: Float in [1, -1] {
                            var turn = SIMD3<Float>(0, 0, 0)
                            turn[axis] = sign * stepDeg * .pi / 180
                            let candidate = best * smallRotation(turn)
                            let s = ncc(candidate, level)
                            if s > score + 1e-5 {
                                best = candidate
                                score = s
                                improved = true
                            }
                        }
                    }
                }
            }
        }
        return (simd_float3x3(simd_quatf(best)), score,
                coarse[0].score, margin)
    }

    /// Weighted normalised cross-correlation between the reference
    /// detail and the SDK panorama's detail at rotation · direction.
    private static func ncc(_ M: simd_float3x3, _ level: Level) -> Float {
        let w = level.w, h = level.h
        let W = Float(w), H = Float(h)
        var sw = 0.0, sa = 0.0, sb = 0.0
        var saa = 0.0, sbb = 0.0, sab = 0.0
        level.pano.withUnsafeBufferPointer { p in
            for i in 0..<level.dirs.count {
                let d = M * level.dirs[i]
                let lon = atan2(d.x, d.z)
                let lat = asin(max(-1, min(1, -d.y)))
                // The SDK panorama, shrunk by ImageIO and then halved,
                // has its pixel centres at +0.5.
                var x = (lon / (2 * .pi) + 0.5) * W - 0.5
                if x < 0 { x += W }
                if x >= W { x -= W }
                let y = min(max((0.5 - lat / .pi) * H - 0.5, 0), H - 1)
                let x0 = min(Int(x), w - 1), y0 = min(Int(y), h - 1)
                let x1 = x0 + 1 < w ? x0 + 1 : 0
                let y1 = min(y0 + 1, h - 1)
                let fx = x - Float(x0), fy = y - Float(y0)
                let top = p[y0 * w + x0]
                    + (p[y0 * w + x1] - p[y0 * w + x0]) * fx
                let bot = p[y1 * w + x0]
                    + (p[y1 * w + x1] - p[y1 * w + x0]) * fx
                let b = Double(top + (bot - top) * fy)
                let a = Double(level.ref[i])
                let wt = Double(level.weight[i])
                sw += wt
                sa += wt * a
                sb += wt * b
                saa += wt * a * a
                sbb += wt * b * b
                sab += wt * a * b
            }
        }
        guard sw > 0 else { return 0 }
        let ma = sa / sw, mb = sb / sw
        let va = saa / sw - ma * ma, vb = sbb / sw - mb * mb
        guard va > 1e-9, vb > 1e-9 else { return 0 }
        return Float((sab / sw - ma * mb) / (va * vb).squareRoot())
    }

    /// The 24 rotations that map the axes onto the axes.
    private static func rightAngleRotations() -> [simd_float3x3] {
        let perms = [[0, 1, 2], [0, 2, 1], [1, 0, 2],
                     [1, 2, 0], [2, 0, 1], [2, 1, 0]]
        var out: [simd_float3x3] = []
        for p in perms {
            for signs in 0..<8 {
                var r = [SIMD3<Float>](repeating: .zero, count: 3)
                for i in 0..<3 {
                    r[i][p[i]] = (signs >> i) & 1 == 1 ? -1 : 1
                }
                let m = simd_float3x3(rows: r)
                if simd_determinant(m) > 0 { out.append(m) }
            }
        }
        return out
    }

    private static func smallRotation(_ turn: SIMD3<Float>) -> simd_float3x3 {
        let angle = simd_length(turn)
        guard angle > 1e-9 else { return matrix_identity_float3x3 }
        return simd_float3x3(simd_quatf(angle: angle, axis: turn / angle))
    }

    // MARK: - Image helpers

    /// Image minus its 7×7 box blur: the fine detail, without the
    /// brightness and colour differences between the two stitchers.
    /// Wraps horizontally (the panorama does), clamps vertically.
    private static func detail(_ g: [Float], w: Int, h: Int) -> [Float] {
        let r = 3
        var rowSum = [Float](repeating: 0, count: w * h)
        for y in 0..<h {
            let row = y * w
            var acc: Float = 0
            for k in -r...r { acc += g[row + ((k % w) + w) % w] }
            for x in 0..<w {
                rowSum[row + x] = acc
                acc += g[row + (x + r + 1) % w]
                    - g[row + (x - r + w) % w]
            }
        }
        var out = [Float](repeating: 0, count: w * h)
        let n = Float((2 * r + 1) * (2 * r + 1))
        for x in 0..<w {
            var acc: Float = 0
            for k in -r...r { acc += rowSum[min(max(k, 0), h - 1) * w + x] }
            for y in 0..<h {
                out[y * w + x] = g[y * w + x] - acc / n
                acc += rowSum[min(y + r + 1, h - 1) * w + x]
                    - rowSum[max(y - r, 0) * w + x]
            }
        }
        return out
    }

    /// Half size, each pixel the mean of a 2×2 block.
    private static func half(_ g: [Float], w: Int, h: Int) -> [Float] {
        let w2 = w / 2, h2 = h / 2
        var out = [Float](repeating: 0, count: w2 * h2)
        for y in 0..<h2 {
            for x in 0..<w2 {
                let i = 2 * y * w + 2 * x
                out[y * w2 + x] = 0.25 * (g[i] + g[i + 1] + g[i + w] + g[i + w + 1])
            }
        }
        return out
    }

    private static func grayFromRGBA(_ px: [UInt8], count: Int) -> [Float] {
        var g = [Float](repeating: 0, count: count)
        for i in 0..<count {
            let o = i * 4
            g[i] = 0.299 * Float(px[o]) + 0.587 * Float(px[o + 1])
                + 0.114 * Float(px[o + 2])
        }
        return g
    }

    /// The image at url, resized to width × height, as gray levels with
    /// the top row first.
    private static func loadGray(
        url: URL, width: Int, height: Int
    ) -> [Float]? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateThumbnailAtIndex(source, 0, [
                kCGImageSourceThumbnailMaxPixelSize: max(width, height),
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceShouldCacheImmediately: true,
              ] as CFDictionary)
        else { return nil }
        var px = [UInt8](repeating: 0, count: width * height)
        let drew = px.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(
                data: buf.baseAddress, width: width, height: height,
                bitsPerComponent: 8, bytesPerRow: width,
                space: CGColorSpaceCreateDeviceGray(),
                bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return false }
            ctx.interpolationQuality = .high
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        return drew ? px.map { Float($0) } : nil
    }

    // MARK: - Rotation helpers

    private static func angleDeg(_ m: simd_float3x3) -> Float {
        let c = (m[0][0] + m[1][1] + m[2][2] - 1) / 2
        return acos(max(-1, min(1, c))) * 180 / .pi
    }

    private static func rows(_ m: simd_float3x3) -> [Float] {
        (0..<3).flatMap { r in (0..<3).map { c in m[c][r] } }
    }

    private static func matrix(rows a: [Float]) -> simd_float3x3 {
        simd_float3x3(rows: [
            SIMD3(a[0], a[1], a[2]),
            SIMD3(a[3], a[4], a[5]),
            SIMD3(a[6], a[7], a[8]),
        ])
    }
}
