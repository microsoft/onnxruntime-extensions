package ai.onnxruntime.extensions;


public class OrtxPackage implements AutoCloseable {

    public String getLibraryPath() {
        return OrtxLibrary.getExtractedLibraryPath();
    }

    @Override
    public void close() {

    }
}
