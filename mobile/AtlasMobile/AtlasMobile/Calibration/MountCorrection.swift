import Foundation
import simd

/// A small turn of the 360 camera on its mount, measured once with an SfM
/// alignment and applied by the export (COLMAPExporter.erpFromARKit) on
/// top of Camera Settings' Mount Rotation.
///
/// The default is session 2026-11-15_17-52-39 (X6 on the user's mount),
/// aligned in Spirula Studio: with the phone's poses at command + 1.0 s,
/// all 28 lens images were turned by the same 4.06° (front lens 4.06°,
/// back lens 3.93°, the 0.5° between them being the lens rig). Removing
/// it took the error from a median 4.6° to 1.6°. The camera positions
/// already matched to 3 cm. Shutter delays from 0 to 4 s fitted equally
/// well, so the delay was left at 1.0 s.
///
/// The value is a rotation vector in degrees in the front lens's camera
/// frame (COLMAP: x right, y down, z forward). It belongs to the mount: a
/// re-mounted camera needs a new measurement or the correction off.
enum MountCorrection {
    static let enabledKey = "mountCorrectionEnabled"
    static let xKey = "mountCorrectionX"
    static let yKey = "mountCorrectionY"
    static let zKey = "mountCorrectionZ"

    static let measured = SIMD3<Double>(2.408, 3.028, -1.218)
    static let measuredNote =
        "Measured on session 2026-11-15_17-52-39 (X6): 4.06° turn, "
        + "360 view error 4.6° → 1.6°."

    /// Registers the measured value as the default, so the correction
    /// is on until it is turned off in Camera Settings.
    static func registerDefaults() {
        UserDefaults.standard.register(defaults: [
            enabledKey: true,
            xKey: measured.x, yKey: measured.y, zKey: measured.z,
        ])
    }

    /// The rotation vector in degrees, as Camera Settings holds it.
    static var degrees: SIMD3<Double> {
        registerDefaults()
        let d = UserDefaults.standard
        return SIMD3(d.double(forKey: xKey), d.double(forKey: yKey),
                     d.double(forKey: zKey))
    }

    /// The correction to apply, or nil when it is off or zero.
    static var current: simd_float3x3? {
        registerDefaults()
        guard UserDefaults.standard.bool(forKey: enabledKey) else {
            return nil
        }
        return matrix(degrees)
    }

    static func angleDegrees(_ v: SIMD3<Double>) -> Double {
        simd_length(v)
    }

    static func matrix(_ v: SIMD3<Double>) -> simd_float3x3? {
        let deg = simd_length(v)
        guard deg > 1e-6, deg.isFinite else { return nil }
        let axis = SIMD3<Float>(simd_normalize(v))
        let q = simd_quatf(angle: Float(deg * .pi / 180), axis: axis)
        return simd_float3x3(q)
    }

    /// One line for the export trace.
    static var note: String {
        guard current != nil else { return "mount correction: off" }
        let v = degrees
        return String(
            format: "mount correction: %.2f° (rotation vector %.3f, %.3f,"
                + " %.3f° in the front lens frame)",
            angleDegrees(v), v.x, v.y, v.z)
    }
}
