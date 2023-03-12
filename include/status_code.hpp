#ifndef WBEE_INFER_STATUS_CODE_HPP
#define WBEE_INFER_STATUS_CODE_HPP
namespace wbee_infer {
    enum class RuntimeParameterType {
        kParameterUnknown = 0,
        kParameterBool = 1,
        kParameterInt = 2,

        kParameterFloat = 3,
        kParameterString = 4,
        kParameterIntArray = 5,
        kParameterFloatArray = 6,
        kParameterStringArray = 7,      
    };
}
#endif