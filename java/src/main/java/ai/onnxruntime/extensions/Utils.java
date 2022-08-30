package ai.onnxruntime.extensions;


public class Utils {
    public static String getExtensionVersion() {
        return "1.0.0";
    }

    private static native long getNativeExtensionOperatorRegister();
}
