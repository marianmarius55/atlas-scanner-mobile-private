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

    func makeCoordinator() -> Coordinator {
        Coordinator(coverage: coverage)
    }

    func makeUIView(context: Context) -> ARSCNView {
        let scnView = ARSCNView()
        scnView.session = session
        scnView.automaticallyUpdatesLighting = true
        scnView.rendersCameraGrain = true
        scnView.contentMode = .scaleAspectFill
        scnView.delegate = context.coordinator
        // The root node is ARKit's world frame, the frame of the
        // coverage map's points.
        scnView.scene.rootNode.addChildNode(
            context.coordinator.coverageNode)
        return scnView
    }

    func updateUIView(_ uiView: ARSCNView, context: Context) {
        let coord = context.coordinator
        coord.showMesh = showMesh
        coord.coverageNode.isHidden = !showCoverage

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
        private let coverage: CoverageMap
        /// Version of the snapshot coverageNode shows. Render thread only.
        private var coverageVersion = -1

        init(coverage: CoverageMap) {
            self.coverage = coverage
            let node = SCNNode()
            node.name = "coverage"
            node.isHidden = true
            self.coverageNode = node
            super.init()
        }

        /// Swaps in the new point cloud after the map changes (once per
        /// saved scan). Called on SceneKit's render thread before each
        /// frame, the place for scene changes.
        func renderer(
            _ renderer: SCNSceneRenderer,
            updateAtTime time: TimeInterval
        ) {
            let snap = coverage.snapshot()
            guard snap.version != coverageVersion else { return }
            coverageVersion = snap.version
            coverageNode.geometry = Self.pointCloud(snap)
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

        /// One unlit dot per 5 cm cell, coloured per cell.
        private static func pointCloud(
            _ snap: CoverageMap.Snapshot
        ) -> SCNGeometry? {
            guard snap.count > 0 else { return nil }
            let floatSize = MemoryLayout<Float>.size
            let vertices = SCNGeometrySource(
                data: snap.positions,
                semantic: .vertex,
                vectorCount: snap.count,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: floatSize,
                dataOffset: 0,
                dataStride: 3 * floatSize
            )
            let colors = SCNGeometrySource(
                data: snap.colors,
                semantic: .color,
                vectorCount: snap.count,
                usesFloatComponents: true,
                componentsPerVector: 3,
                bytesPerComponent: floatSize,
                dataOffset: 0,
                dataStride: 3 * floatSize
            )
            let element = SCNGeometryElement(
                data: snap.indices,
                primitiveType: .point,
                primitiveCount: snap.count,
                bytesPerIndex: MemoryLayout<UInt32>.size
            )
            // Cell-sized dots, clamped on screen: near walls get a dot
            // grid you can see through, far ones stay visible.
            element.pointSize = CGFloat(CoverageMap.cellSize)
            element.minimumPointScreenSpaceRadius = 2
            element.maximumPointScreenSpaceRadius = 8

            let geometry = SCNGeometry(
                sources: [vertices, colors], elements: [element])
            let material = SCNMaterial()
            material.lightingModel = .constant
            material.diffuse.contents = UIColor.white
            geometry.materials = [material]
            return geometry
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
