# How to start

1. correct the onnxruntime-extensions JAR location if needed, which is located at app/build.gradle: 16


    `implementation fileTree(dir: '..\\..\\..\\out\\Windows\\java\\build\\libs', include: ['onnxruntime-extensions-0.*.0.jar'])
`

2. build and run this java project.
