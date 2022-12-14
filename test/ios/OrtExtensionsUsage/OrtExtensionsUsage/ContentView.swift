// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import SwiftUI

struct ContentView: View {
    func runOrtDecodeAndCheckImage() -> String {
        do {
            try OrtClient.decodeAndCheckImage()
            return "Ok"
        } catch let error as NSError {
            return "Error: \(error.localizedDescription)"
        }
    }

    var body: some View {
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundColor(.accentColor)
            Text("Decode image result: \(runOrtDecodeAndCheckImage())")
                .accessibilityIdentifier("decodeImageResult")
        }
        .padding()
    }
}

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
