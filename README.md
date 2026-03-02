## Тензорный компилятор

Устновить и собрать проект:
```
git clone https://github.com/aywert/TensorCompiler
cd TensorCompiler
cmake -S . -B build
cmake --build build
```

Чтобы исполнить программу на вход необходимо подать файл с расширением ```.onnx```. Файлы с таким расширением можно найти в папки ```examples```

Пример:
```
./build/tensor ./examples/conv_layer.onnx
```
Результатом работы программы является файл ```graph.got```, который находится в папке ```grapviz```. 
Для того, чтобы получить изображение графа запустите:
```
cd graphviz
dot -Tpng graph.dot -o graph.png
```
Файл формата png следует искать в той же папке ```grapviz```.

## Тестировка (unit tests)
Запустить тесты из корневой папку проекта:
```
cd build
ctest
```

