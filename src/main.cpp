#include <iostream>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/message.h>
#include "onnx.pb.h"


int main() {
    // Создаем экземпляр сообщения ModelProto
    onnx::ModelProto model;

    
    // Устанавливаем версию IR (просто для теста)
    model.set_ir_version(onnx::IR_VERSION);

    std::cout << "Protobuf library successfully linked!" << std::endl;
    std::cout << "Current ONNX IR Version: " << model.ir_version() << std::endl;

    return 0;
}