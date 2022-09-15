package ai.onnxruntime.extensions;


public final class OrtxPackage implements AutoCloseable {

    private static volatile OrtxPackage INSTANCE;

    public String getLibraryPath() {
        return OrtxLibrary.getExtractedLibraryPath();
    }

    public static synchronized OrtxPackage getPackage() {
        if  (INSTANCE == null) {
            return new OrtxPackage();
        }
        else {
            return INSTANCE;
        }
    }

    @Override
    public void close() {
        // Reserved
    }
}
