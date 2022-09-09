package ai.onnxruntime.extensions;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;


public class OrtxLibrary {

    private static File tempLibraryFile;

    static{
        try{
            String path = "/ai/onnxruntime/extensions/native/linux-x64/" + System.mapLibraryName("onnxruntime_extensions4j_jni");
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
