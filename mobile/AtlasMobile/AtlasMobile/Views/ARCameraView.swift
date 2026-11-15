import SwiftUI
import ARKit
import SceneKit

struct ARCameraView: UIViewRepresentable {
    let session: ARSession
    var showMesh: Bool
    var showPointCloud: Bool
    /// LiDAR coverage of the session, drawn as dots when showCoverage
    /// is on.
    let coverage: CoverageMap
    var showCoverage: Bool
    /// 360 shot guide, drawn when showGuide is on: globes where to stand
    /// and at which heights, and the room's cells coloured by how many
    /// spots see them.
    let guide: PanoramaGuide
    var showGuide: Bool

    func makeCoordinator() -> Coordinator {
        Coordinator(coverage: coverage, guide: guide)
    }

    func makeUIView(context: Context) -> ARSCNView {
        let scnView = ARSCNView()
        scnView.session = session
        scnView.automaticallyUpdatesLighting = true
        scnView.rendersCameraGrain = true
        scnView.contentMode = .scaleAspectFill
        scnView.delegate = context.coordinator
        // The root node is ARKit's world frame, the frame of the
        // coverage map's points and the guide's spots.
        scnView.scene.rootNode.addChildNode(
            context.coordinator.coverageNode)
        scnView.scene.rootNode.addChildNode(
            context.coordinator.guideNode)
        return scnView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        let coord = context.coordinator
        coord.showMesh = showMesh
        coord.coverageNode.isHidden = !showCoverage
        coord.guideNode.isHidden = !showGuide

        if showPointCloud {
            uiView.debugOptions.insert(.showFeaturePoints)
        } else {
            uiView.debugOptions.remove(.showFeaturePoints)
        }

        uiView.scene.rootNode.enumerateChildNodes { node, _ in
            if node.name == "mesh" {
                node.isHidden = !showMesh
            }
        }
    }

    final class Coordinator: NSObject, ARSCNViewDelegate {
        var showMesh = false
        /// The coverage map's cells as points.
        let coverageNode: SCNNode
        /// The guide: its dots and its spot markers.
        let guideNode: SCNNode
        private let guideDots: SCNNode
        private let guideMarkers: SCNNode
        private let coverage: CoverageMap
        private let guide: PanoramaGuide
        // Versions of what the nodes show. Render thread only.
        private var coverageVersion = -1
        private var guideVersion = -1
        private var guideDotsVersion = -1

        init(coverage: CoverageMap, guide: PanoramaGuide) {
            self.coverage = coverage
            self.guide = guide
            let node = SCNNode()
            node.name = "coverage"
            node.isHidden = true
            self.coverageNode = node
            let dots = SCNNode()
            let markers = SCNNode()
            let guideRoot = SCNNode()
            guideRoot.name = "guide"
            guideRoot.isHidden = true
            guideRoot.addChildNode(dots)
            guideRoot.addChildNode(markers)
            self.guideDots = dots
            self.guideMarkers = markers
            self.guideNode = guideRoot
            super.init()
        }

        /// Swaps in new dots and markers when the coverage map or the guide
        /// change. Called on SceneKit's render thread before each frame,
        /// the place for scene changes.
        func renderer(
            _ renderer: SCNSceneRenderer,
            updateAtTime time: TimeInterval
        ) {
            let snap = coverage.snapshot()
            if snap.version != coverageVersion {
                coverageVersion = snap.version
                coverageNode.geometry = Self.pointCloud(
                    positions: snap.positions, colors: snap.colors,
                    indices: snap.indices, count: snap.count,
                    pointSize: CoverageMap.cellSize)
            }
            let plan = guide.snapshot()
            if plan.dotsVersion != guideDotsVersion {
                guideDotsVersion = plan.dotsVersion
                guideDots.geometry = Self.pointCloud(
                    positions: plan.positions, colors: plan.colors,
                    indices: plan.indices, count: plan.dotCount,
                    pointSize: PanoramaGuide.planCell)
            }
            if plan.version != guideVersion {
                guideVersion = plan.version
                rebuildMarkers(plan)
            }
        }

        func renderer(
            _ renderer: SCNSceneRenderer,
            didAdd node: SCNNode,
            for anchor: ARAnchor
        ) {
            guard showMesh, let meshAnchor = anchor as? ARMeshAnchor else { return }
            node.addChildNode(buildMeshNode(from: meshAnchor))
        }

        func renderer(
            _ renderer: SCNSceneRenderer,
            didUpdate node: SCNNode,
            for anchor: ARAnchor
        ) {
            guard let meshAnchor = anchor as? ARMeshAnchor else { return }
            node.childNodes.filter { $0.name == "mesh" }.forEach { $0.removeFromParentNode() }
            if showMesh {
                node.addChildNode(buildMeshNode(from: meshAnchor))
            }
        }

        /// One unlit dot per cell, coloured per cell.
        private static func pointCloud(
            positions: Data, colors: Data, indices: Data,
            count: Int, pointSize: Float
        ) -> SCNGeometry? {
            guard count > 0 else { return nil }
            let floatSize = MemoryLayout<Float>.size
            let vertices = SCNGeometrySource(
                data: positions,
                semantic: .vertex,
                vectorCount: count,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: floatSize,
                dataOffset: 0,
                dataStride: 3 * floatSize
            )
            let colorSource = SCNGeometrySource(
                data: colors,
                semantic: .color,
                vectorCount: count,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: floatSize,
                dataOffset: 0,
                dataStride: 3 * floatSize
            )
            let element = SCNGeometryElement(
                data: indices,
                primitiveType: .point,
                primitiveCount: count,
                bytesPerIndex: MemoryLayout<UInt32>.size
            )
            // Cell-sized dots, clamped on screen: near walls get a dot
            // grid you can see through, far ones stay visible.
            element.pointSize = CGFloat(pointSize)
            element.minimumPointScreenSpaceRadius = 2
            element.maximumPointScreenSpaceRadius = 8

            let geometry = SCNGeometry(
                sources: [vertices, colorSource], elements: [element])
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = UIColor.white
            geometry.materials = [material]
            return geometry
        }

        // MARK: - Guide markers

        private static let targetColor = UIColor(
            red: 0.0, green: 0.85, blue: 1.0, alpha: 1)
        private static let doneColor = UIColor(
            red: 0.2, green: 0.85, blue: 0.4, alpha: 1)
        private static let failedColor = UIColor(
            red: 1.0, green: 0.3, blue: 0.25, alpha: 1)

        private func rebuildMarkers(_ plan: PanoramaGuide.Snapshot) {
            for child in guideMarkers.childNodes {
                child.removeFromParentNode()
            }
            if let p = plan.pending {
                guideMarkers.addChildNode(Self.pendingNode(at: p))
            }
            guard let floorY = plan.floorY else { return }
            for spot in plan.spots {
                guideMarkers.addChildNode(
                    Self.spotNode(spot, floorY: floorY))
            }
            if let target = plan.target {
                guideMarkers.addChildNode(
                    Self.targetNode(target, floorY: floorY))
            }
        }

        /// A spot with shots: a ring on the floor (green when complete,
        /// yellow when started, red when only failed), a pole, and a ball
        /// per level: green taken, red moved (retake), white still to do.
        private static func spotNode(
            _ spot: PanoramaGuide.SpotMarker, floorY: Float
        ) -> SCNNode {
            let node = SCNNode()
            let done = spot.states.filter { $0 == .done }.count
            let ringColor: UIColor = spot.isComplete
                ? doneColor : (done > 0 ? .systemYellow : failedColor)
            node.addChildNode(ring(
                at: spot.center, floorY: floorY, radius: 0.18,
                pipe: 0.006, color: ringColor))
            node.addChildNode(pole(
                at: spot.center, floorY: floorY,
                top: spot.levelY.max() ?? floorY,
                color: UIColor.white.withAlphaComponent(0.35)))
            for (l, state) in spot.states.enumerated() {
                let color: UIColor
                switch state {
                case .done: color = doneColor
                case .failed: color = failedColor
                case .todo: color = UIColor.white.withAlphaComponent(0.55)
                case .blocked: continue
                }
                let ball = SCNNode(geometry: sphere(
                    radius: 0.045, color: color, wireframe: false))
                ball.position = SCNVector3(
                    spot.center.x, spot.levelY[l], spot.center.y)
                node.addChildNode(ball)
            }
            return node
        }

        /// Where to go: a pulsing ring on the floor and a globe at each
        /// level still to shoot; the next one is bigger and pulses.
        private static func targetNode(
            _ target: PanoramaGuide.SpotMarker, floorY: Float
        ) -> SCNNode {
            let node = SCNNode()
            let floorRing = ring(
                at: target.center, floorY: floorY, radius: 0.3,
                pipe: 0.012, color: targetColor)
            floorRing.runAction(SCNAction.repeatForever(SCNAction.sequence([
                SCNAction.fadeOpacity(to: 0.35, duration: 0.7),
                SCNAction.fadeOpacity(to: 1.0, duration: 0.7),
            ])))
            node.addChildNode(floorRing)
            node.addChildNode(pole(
                at: target.center, floorY: floorY,
                top: target.levelY.max() ?? floorY,
                color: targetColor.withAlphaComponent(0.5)))
            let next = target.nextLevel
            for (l, state) in target.states.enumerated()
            where state == .todo || state == .failed {
                let isNext = l == next
                let globe = SCNNode(geometry: sphere(
                    radius: isNext ? 0.16 : 0.1,
                    color: targetColor.withAlphaComponent(isNext ? 0.9 : 0.45),
                    wireframe: true))
                globe.position = SCNVector3(
                    target.center.x, target.levelY[l], target.center.y)
                globe.runAction(SCNAction.repeatForever(SCNAction.rotateBy(
                    x: 0, y: CGFloat.pi * 2, z: 0, duration: 8)))
                if isNext {
                    let grow = SCNAction.scale(to: 1.2, duration: 0.6)
                    grow.timingMode = .easeInEaseOut
                    let shrink = SCNAction.scale(to: 1.0, duration: 0.6)
                    shrink.timingMode = .easeInEaseOut
                    globe.runAction(SCNAction.repeatForever(
                        SCNAction.sequence([grow, shrink])))
                }
                node.addChildNode(globe)
            }
            return node
        }

        /// A shot in flight: a white ball where it was sent from.
        private static func pendingNode(at p: SIMD3<Float>) -> SCNNode {
            let ball = SCNNode(geometry: sphere(
                radius: 0.06, color: UIColor.white.withAlphaComponent(0.8),
                wireframe: false))
            ball.position = SCNVector3(p.x, p.y, p.z)
            ball.runAction(SCNAction.repeatForever(SCNAction.sequence([
                SCNAction.fadeOpacity(to: 0.3, duration: 0.3),
                SCNAction.fadeOpacity(to: 1.0, duration: 0.3),
            ])))
            return ball
        }

        private static func ring(
            at center: SIMD2<Float>, floorY: Float, radius: CGFloat,
            pipe: CGFloat, color: UIColor
        ) -> SCNNode {
            // A torus lies in its x-z plane: flat on the floor.
            let torus = SCNTorus(ringRadius: radius, pipeRadius: pipe)
            torus.materials = [material(color, wireframe: false)]
            let node = SCNNode(geometry: torus)
            node.position = SCNVector3(center.x, floorY + 0.01, center.y)
            return node
        }

        private static func pole(
            at center: SIMD2<Float>, floorY: Float, top: Float,
            color: UIColor
        ) -> SCNNode {
            let height = max(0.05, top - floorY)
            let cylinder = SCNCylinder(radius: 0.004, height: CGFloat(height))
            cylinder.materials = [material(color, wireframe: false)]
            let node = SCNNode(geometry: cylinder)
            node.position = SCNVector3(
                center.x, floorY + height / 2, center.y)
            return node
        }

        private static func sphere(
            radius: CGFloat, color: UIColor, wireframe: Bool
        ) -> SCNSphere {
            let s = SCNSphere(radius: radius)
            // Few segments: in wireframe the lines read as a globe's grid.
            s.segmentCount = wireframe ? 16 : 24
            s.materials = [material(color, wireframe: wireframe)]
            return s
        }

        private static func material(
            _ color: UIColor, wireframe: Bool
        ) -> SCNMaterial {
            let m = SCNMaterial()
            m.lightingModel = .constant
            m.diffuse.contents = color
            m.isDoubleSided = true
            if wireframe {
                m.fillMode = .lines
            }
            return m
        }

        private func buildMeshNode(from meshAnchor: ARMeshAnchor) -> SCNNode {
            let geo = meshAnchor.geometry

            let vertexSource = SCNGeometrySource(
                buffer: geo.vertices.buffer,
                vertexFormat: geo.vertices.format,
                semantic: .vertex,
                vertexCount: geo.vertices.count,
                dataOffset: geo.vertices.offset,
                dataStride: geo.vertices.stride
            )

            let faceElement = SCNGeometryElement(
                buffer: geo.faces.buffer,
                primitiveType: .triangles,
                primitiveCount: geo.faces.count,
                bytesPerIndex: geo.faces.bytesPerIndex
            )

            let scnGeo = SCNGeometry(sources: [vertexSource], elements: [faceElement])
            let material = SCNMaterial()
            material.fillMode = .lines
            material.diffuse.contents = UIColor.cyan.withAlphaComponent(0.5)
            material.isDoubleSided = true
            scnGeo.materials = [material]

            let node = SCNNode(geometry: scnGeo)
            node.name = "mesh"
            return node
        }
    }
}
