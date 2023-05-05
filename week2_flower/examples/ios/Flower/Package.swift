// swift-tools-version: 5.6
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Flower",
    products: [
        // Products define the executables and libraries a package produces, and make them visible to other packages.
        .library(
            name: "Flower",
            targets: ["Flower"]),
    ],
    dependencies: [
        // Dependencies declare other packages that this package depends on.
        // .package(url: /* package url */, from: "1.0.0"),
        .package(url: "https://github.com/pvieito/PythonKit.git", branch: "master"),
        .package(url: "https://github.com/kewlbear/NumPy-iOS.git", branch: "main"),
        .package(url: "https://github.com/grpc/grpc-swift.git", from: "1.0.0"),
    ],
    targets: [
        // Targets are the basic building blocks of a package. A target can define a module or a test suite.
        // Targets can depend on other targets in this package, and on products in packages this package depends on.
        .target(
            name: "Flower",
            dependencies: [
                .product(name: "GRPC", package: "grpc-swift"),
                .product(name: "NumPy-iOS", package: "NumPy-iOS"),
                .product(name: "PythonKit", package: "PythonKit")],
            path: "Sources/Flower"),
        .testTarget(
            name: "FlowerTests",
            dependencies: ["Flower"]),
    ]
)
