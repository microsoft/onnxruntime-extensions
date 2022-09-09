package ai.onnxruntime.extensions;

import java.io.IOException;
import java.io.*;
import java.nio.file.FileSystemNotFoundException;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.ProviderNotFoundException;
import java.nio.file.StandardCopyOption;


public class Utils {
    static{
        try{
            String path = "/ai/onnxruntime/extensions/native/linux-x64/" + System.mapLibraryName("onnxruntime_extensions4j_jni");
            // Obtain filename from path
            String[] parts = path.split("/");
            String filename = parts[parts.length - 1];
    
            // Prepare temporary file
            File temporaryDir = createTempDirectory("ortx4j");
            temporaryDir.deleteOnExit();

            File temp = new File(temporaryDir, filename);

            try (InputStream is = Utils.class.getResourceAsStream(path)) {
                Files.copy(is, temp.toPath(), StandardCopyOption.REPLACE_EXISTING);
            } catch (IOException e) {
                temp.delete();
                throw e;
            } catch (NullPointerException e) {
                temp.delete();
                throw new FileNotFoundException("File " + path + " was not found inside JAR.");
            }

            try {
                System.load(temp.getAbsolutePath());
            } finally {
                if (isPosixCompliant()) {
                    // Assume POSIX compliant file system, can be deleted after loading
                    temp.delete();
                } else {
                    // Assume non-POSIX, and don't delete until last file descriptor closed
                    temp.deleteOnExit();
                }
            }
        } catch(IOException e1){
            throw new RuntimeException(e1);
        }
    }

    public static String getExtensionVersion() {
        return "1.0.0";
    }

    private static boolean isPosixCompliant() {
        try {
            return FileSystems.getDefault()
                    .supportedFileAttributeViews()
                    .contains("posix");
        } catch (FileSystemNotFoundException
                | ProviderNotFoundException
                | SecurityException e) {
            return false;
        }
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
