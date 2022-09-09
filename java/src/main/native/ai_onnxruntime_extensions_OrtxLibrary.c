#include "ai_onnxruntime_extensions_OrtxLibrary.h"
#include "onnxruntime_extensions.h"

/*
 * Class:     ai_onnxruntime_extensions_OrtxLibrary
 * Method:    getNativeExtensionOperatorRegister
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_extensions_Utils_getNativeExtensionOperatorRegister(JNIEnv* env, jclass cls) {
  return (jlong)(&RegisterCustomOps);
}
