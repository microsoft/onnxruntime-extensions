// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import XCTest
@testable import OrtExtensionsMacOSUsage

final class OrtExtensionsMacOSUsageTests: XCTestCase {

    override func setUpWithError() throws {
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }

    override func tearDownWithError() throws {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
    }

    func testDecodeAndCheckImage() throws {
      // test that it doesn't throw
      try OrtClient.decodeAndCheckImage()
    }

    func testSwiftDecodeAndCheckImage() throws {
      // test that it doesn't throw
      try swiftDecodeAndCheckImage()
    }
}
