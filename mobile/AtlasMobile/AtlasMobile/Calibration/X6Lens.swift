import Foundation
import simd

/// One lens of a dual-fisheye photo as COLMAP's THIN_PRISM_FISHEYE, the
/// camera model Spirula Studio's 360-camera preset gives each lens of a 360
/// camera (src/app/gui/DatasetPreset.cpp). A ray at theta from the lens
/// axis, in direction (ux, uy) across the image (x right, y down), is at
/// (u, v) = theta (ux, uy), distorted to
///   u' = u (1 + k1 r2 + k2 r4 + k3 r6 + k4 r8) + 2 p1 u v + p2 (r2 + 2 u u) + sx1 r2
///   v' = v (1 + k1 r2 + ...) + 2 p2 u v + p1 (r2 + 2 v v) + sy1 r2
/// with r2 = u u + v v, and lands at (fx u', fy v') + (cx, cy). fx, fy, cx
/// and cy are fractions of the lens image's height, in COLMAP's pixel
/// convention (the first pixel's centre at 0.5); cx is within the lens's
/// own half of the photo.
struct ThinPrismLens {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let k1: Double, k2: Double, p1: Double, p2: Double
    let k3: Double, k4: Double, sx1: Double, sy1: Double
    /// Rays further than this from the axis (radians) are outside the lens.
    let maxTheta: Double
    let maxThetaF: Float
    // The same numbers in single precision, for sampling pixel by pixel.
    private let fxs: Float, fys: Float, cxs: Float, cys: Float
    private let k1s: Float, k2s: Float, k3s: Float, k4s: Float
    private let p1s: Float, p2s: Float, sx1s: Float, sy1s: Float

    init(fx: Double, fy: Double, cx: Double, cy: Double,
         k1: Double = 0, k2: Double = 0, p1: Double = 0, p2: Double = 0,
         k3: Double = 0, k4: Double = 0, sx1: Double = 0, sy1: Double = 0,
         maxTheta: Double) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.k1 = k1
        self.k2 = k2
        self.p1 = p1
        self.p2 = p2
        self.k3 = k3
        self.k4 = k4
        self.sx1 = sx1
        self.sy1 = sy1
        self.maxTheta = maxTheta
        maxThetaF = Float(maxTheta)
        fxs = Float(fx)
        fys = Float(fy)
        cxs = Float(cx)
        cys = Float(cy)
        k1s = Float(k1)
        k2s = Float(k2)
        k3s = Float(k3)
        k4s = Float(k4)
        p1s = Float(p1)
        p2s = Float(p2)
        sx1s = Float(sx1)
        sy1s = Float(sy1)
    }

    /// COLMAP's THIN_PRISM_FISHEYE params, `params` in pixels of a lens
    /// image `size` pixels high, as a fraction of it.
    init(colmap p: [Double], size: Double, maxThetaDeg: Double) {
        self.init(fx: p[0] / size, fy: p[1] / size,
                  cx: p[2] / size, cy: p[3] / size,
                  k1: p[4], k2: p[5], p1: p[6], p2: p[7],
                  k3: p[8], k4: p[9], sx1: p[10], sy1: p[11],
                  maxTheta: maxThetaDeg * .pi / 180)
    }

    /// The app's radial curve as this model, centred where the stitcher
    /// puts a lens: the middle of its half (pixel halfWidth / 2, height / 2
    /// with pixel centres at whole numbers). The curve's
    /// r = height theta (k1 + k3 theta^2 + k5 theta^4) is
    /// f theta (1 + k1' theta^2 + k2' theta^4) with f = k1.
    init(curve: LensCurve, halfWidth: Int, height: Int) {
        let h = Double(height)
        let k1 = Double(curve.k1)
        self.init(fx: k1, fy: k1,
                  cx: (Double(halfWidth) * 0.5 + 0.5) / h,
                  cy: (h * 0.5 + 0.5) / h,
                  k1: Double(curve.k3) / k1,
                  k2: Double(curve.k5) / k1,
                  maxTheta: Double(curve.maxTheta))
    }

    /// COLMAP's THIN_PRISM_FISHEYE params for a lens image `height` pixels
    /// high: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1.
    func colmapParams(height: Int) -> [Double] {
        let s = Double(height)
        return [fx * s, fy * s, cx * s, cy * s,
                k1, k2, p1, p2, k3, k4, sx1, sy1]
    }

    /// Where a ray lands in a lens image `height` pixels high, in the
    /// stitcher's pixel convention (pixel centres at whole numbers).
    /// `theta` is the ray's angle from the axis, (ux, uy) its unit
    /// direction across the image.
    @inline(__always)
    func pixel(theta: Float, ux: Float, uy: Float,
               height: Float) -> (x: Float, y: Float) {
        let u = theta * ux
        let v = theta * uy
        let r2 = u * u + v * v
        let radial = r2 * (k1s + r2 * (k2s + r2 * (k3s + r2 * k4s)))
        let du = u * radial + 2 * p1s * u * v
            + p2s * (r2 + 2 * u * u) + sx1s * r2
        let dv = v * radial + 2 * p2s * u * v
            + p1s * (r2 + 2 * v * v) + sy1s * r2
        return ((fxs * (u + du) + cxs) * height - 0.5,
                (fys * (v + dv) + cys) * height - 0.5)
    }
}

/// The Insta360 X6's lenses for photos (7760 px a lens), hardcoded for every
/// X6. Measured on session 2026-11-15 13-22-17 (camera BXEA3ABFW6PED9,
/// firmware v1.3.59), 17 photos:
///
/// - Front lens (left half) and both lens centres: the camera's factory
///   calibration (offset_v6), read as telemetry-parser reads it and fitted
///   to THIN_PRISM_FISHEYE within 0.8 px out to 105 degrees.
/// - Back lens (right half): Spirula Studio's SfM (`spirula sfm auto`, the
///   GUI's 360 settings: a dual-fisheye rig, thin-prism-fisheye) on the 34
///   lens images and 107 iPhone photos, all lenses registered, 0.62 px
///   mean reprojection. Its focal matches the factory's to 0.5 px. SfM held
///   the centres at the middle, so they are the factory's.
/// - The angle between the lenses: the same SfM's rig, 0.64 degrees off
///   exactly back to back (the factory's own angles say 0.55).
///
/// The front lens's SfM fit was left out: 3-6% off the factory focal and
/// 3% wider than high, which no real lens is.
enum X6Lens {
    static let size = 7760

    static let front = ThinPrismLens(
        colmap: [2088.06829, 2088.06829, 3852.44998, 3888.24063,
                 0.0848033373, -0.0204406267, 0.000489299564,
                 0.000280557201, 0.00014753571, -0.000234918016,
                 0.000808958277, -0.000581895344],
        size: Double(size), maxThetaDeg: 109.61)

    static let back = ThinPrismLens(
        colmap: [2091.2428, 2092.0283, 3885.10114, 3890.22059,
                 0.092711227, -0.030764425, -0.00073793576,
                 0.0001159225, 0.0048723719, -0.00095990149,
                 -0.00035296622, 0.0019657033],
        size: Double(size), maxThetaDeg: 109.49)

    /// The back lens's camera from the front lens's (COLMAP camera axes,
    /// x right, y down, z forward): Spirula's rigs.txt cam_from_rig for
    /// fisheye_back, quaternion w, x, y, z. Exactly back to back would be
    /// diag(-1, 1, -1).
    static let backFromFront = simd_float3x3(simd_quatf(
        ix: -7.17580830375e-05, iy: 0.999984456367,
        iz: -0.00499645095314, r: -0.00247332850238))

    static func isX6(_ model: String) -> Bool {
        model.lowercased().contains("x6")
    }

    /// Both lenses in a LensCurve (the curve fields are the front lens's
    /// radial part, used only where a single curve is read).
    static var lens: LensCurve {
        var curve = LensCurve(
            k1: Float(front.fx),
            k3: Float(front.fx * front.k1),
            k5: Float(front.fx * front.k2),
            maxTheta: Float(max(front.maxTheta, back.maxTheta)))
        curve.lenses = [front, back]
        curve.note = "the X6's hardcoded lenses (front: factory"
            + " calibration; back: Spirula SfM; lenses 0.64° off back to"
            + " back)"
        return curve
    }
}
