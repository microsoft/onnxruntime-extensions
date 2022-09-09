package ai.onnxruntime.extensions;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


public class UtilsTest {
    @Test
    void getRegisterTest() {
        long handle = Utils.getNativeExtensionOperatorRegister();
        Assertions.assertNotEquals(handle, Long.valueOf(0));
    }
}
