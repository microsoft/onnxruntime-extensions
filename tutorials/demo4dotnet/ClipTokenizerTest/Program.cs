using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Runtime.InteropServices;

namespace ClipTokenizerTestCS
{
    internal class Program
    {
        const string extensionsDllName = "ortextensions";
        const string modelFilePath = "cliptok.onnx";

        static void Main()
        {
            IntPtr handle = NativeLibrary.Load(extensionsDllName);
            if (handle == IntPtr.Zero)
            {
                Console.WriteLine("Microsoft.ML.OnnxRuntime.Extensions nuget package isn't installed");
                return;
            }
            NativeLibrary.Free(handle);
            handle = IntPtr.Zero;

            try
            {
                SessionOptions sessionOptions = new();
                // Register the onnxruntime-extensions DLL for the custom ops.
                sessionOptions.RegisterCustomOpLibraryV2(extensionsDllName, out handle);

                var tokenizeSession = new InferenceSession(modelFilePath, sessionOptions);
                var inputTensor = new DenseTensor<string>(new string[] { "This is a test string for tokenizer" }, new int[] { 1 });
                var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };
                var tokens = tokenizeSession.Run(inputString);
                var ids = tokens.First();
                Console.WriteLine(ids.AsTensor<Int64>().GetArrayString());
            }
            finally { NativeLibrary.Free(handle); }
        }
    }
}
