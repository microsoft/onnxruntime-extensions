package ai.onnxruntime.extensions.apptest;

import android.content.Context;
import android.util.Log;

import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.ext.junit.rules.ActivityScenarioRule;
import androidx.test.ext.junit.runners.AndroidJUnit4;

import org.junit.Test;
import org.junit.runner.RunWith;

import static org.junit.Assert.*;

import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.extensions.OrtxPackage;

import org.junit.Rule;
import org.junit.After;
import com.microsoft.appcenter.espresso.Factory;
import com.microsoft.appcenter.espresso.ReportHelper;

/**
 * Instrumented test, which will execute on an Android device.
 *
 * @see <a href="http://d.android.com/tools/testing">Testing documentation</a>
 */
@RunWith(AndroidJUnit4.class)
public class BertTokenizerTest {

    @Rule
    public ActivityScenarioRule<MainActivity> activityScenarioRule = new ActivityScenarioRule<>(MainActivity.class);

    @Rule
    public ReportHelper reportHelper = Factory.getReportHelper();

    @After
    public void TearDown(){
        reportHelper.label("Stopping App");
    }

    @Test
    public void useAppContext() {
        // Context of the app under test.
        Context appContext = InstrumentationRegistry.getInstrumentation().getTargetContext();
        assertEquals("ai.onnxruntime.extensions.apptest", appContext.getPackageName());
        // TODO: enable verbose logging level here
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        try {
            OrtSession.SessionOptions sess_opt = new OrtSession.SessionOptions();
            /* Register the custom ops from onnxruntime-extensions */
            sess_opt.registerCustomOpLibrary(OrtxPackage.getLibraryPath());
            InputStream modelData = appContext.getResources().openRawResource(R.raw.test_bert_tokenizer);
            ByteArrayOutputStream dataBuffer = new ByteArrayOutputStream();
            byte[] buffer = new byte[1024];
            int readLength;
            while ((readLength = modelData.read(buffer, 0, buffer.length)) != -1) {
                dataBuffer.write(buffer, 0, readLength);
            }
            byte[] bytesModel = dataBuffer.toByteArray();
            OrtSession session = env.createSession(bytesModel, sess_opt);
            OnnxTensor t1 = OnnxTensor.createTensor(env, new String[]{"This is a test"});
            Map<String, OnnxTensor> inputs = new HashMap<String, OnnxTensor>() {{
                put("text", t1);
            }};
            long[] tokenIds;
            try (OrtSession.Result r = session.run(inputs)) {
                tokenIds = (long[])r.get("input_ids").get().getValue();
            }
            long[] expected = {101, 1188, 1110, 170, 2774, 102};
            assertArrayEquals(tokenIds, expected);

        } catch(Exception e1) {
            Log.i("Failed", e1.getMessage());
        }
    }
}
