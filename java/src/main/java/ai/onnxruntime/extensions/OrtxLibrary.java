package ai.onnxruntime.extensions;

import java.io.*;
import java.nio.file.Files;
import java.util.Locale;

import java.nio.file.StandardCopyOption;


public class OrtxLibrary {

    private static File tempLibraryFile;
    private static final String OS_ARCH_STR = getOsArch();
    /**
     * Check if we're running on Android.
     *
     * @return True if the property java.vendor equals The Android Project, false otherwise.
     */
    static boolean isAndroid() {
        return System.getProperty("java.vendor", "generic").equals("The Android Project");
    }

    /* Computes and initializes OS_ARCH_STR (such as linux-x64) */
    private static String getOsArch() {
        String detectedOS = null;
        String os = System.getProperty("os.name", "generic").toLowerCase(Locale.ENGLISH);
        if (os.contains("mac") || os.contains("darwin")) {
        detectedOS = "osx";
        } else if (os.contains("win")) {
        detectedOS = "win";
        } else if (os.contains("nux")) {
        detectedOS = "linux";
        } else if (isAndroid()) {
        detectedOS = "android";
        } else {
        throw new IllegalStateException("Unsupported os:" + os);
        }
        String detectedArch = null;
        String arch = System.getProperty("os.arch", "generic").toLowerCase(Locale.ENGLISH);
        if (arch.startsWith("amd64") || arch.startsWith("x86_64")) {
        detectedArch = "x64";
        } else if (arch.startsWith("x86")) {
        // 32-bit x86 is not supported by the Java API
        detectedArch = "x86";
        } else if (arch.startsWith("aarch64")) {
        detectedArch = "aarch64";
        } else if (arch.startsWith("ppc64")) {
        detectedArch = "ppc64";
        } else if (isAndroid()) {
        detectedArch = arch;
        } else {
        throw new IllegalStateException("Unsupported arch:" + arch);
        }
        return detectedOS + '-' + detectedArch;
    }

    static{
        try{
            String path = "/ai/onnxruntime/extensions/native/" + OS_ARCH_STR
             +"/" + System.mapLibraryName("onnxruntime_extensions4j_jni");
            // Obtain filename from path
            String[] parts = path.split("/");
            String filename = parts[parts.length - 1];
    
            // Prepare temporary file
            File temporaryDir = createTempDirectory("ortx4j");
            temporaryDir.deleteOnExit();

            tempLibraryFile = new File(temporaryDir, filename);

            try (InputStream is = OrtxLibrary.class.getResourceAsStream(path)) {
                Files.copy(is, tempLibraryFile.toPath(), StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                tempLibraryFile.delete();
                throw e;
            } catch (NullPointerException e) {
                tempLibraryFile.delete();
                throw new FileNotFoundException("File " + path + " was not found inside JAR.");
            }

            try {
                System.load(tempLibraryFile.getAbsolutePath());
            } finally {
                tempLibraryFile.deleteOnExit();
            }
        } catch(IOException e1){
            throw new RuntimeException(e1);
        }
    }

    public static String getExtractedLibraryPath() {
        return tempLibraryFile.getAbsolutePath();
    }

    private static File createTempDirectory(String prefix) throws IOException {
        String tempDir = System.getProperty("java.io.tmpdir");
        File generatedDir = new File(tempDir, prefix + System.nanoTime());
        
        if (!generatedDir.mkdir())
            throw new IOException("Failed to create temp directory " + generatedDir.getName());
        
        return generatedDir;
    }

    public static native long getNativeExtensionOperatorRegister();
}
