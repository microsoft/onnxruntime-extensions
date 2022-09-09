package ai.onnxruntime.extensions;

import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;


public class OrtxTest {
    @Test
    void getRegisterTest() {
        long handle = OrtxLibrary.getNativeExtensionOperatorRegister();
        Assertions.assertNotEquals(handle, Long.valueOf(0));
    }
}
