 - The coding style follows the Google coding style. The [c++ guide can be found here](https://google.github.io/styleguide/cppguide.html). Variable names should follow:
    - ```class MyNewClass```
    - ```int some_variable```
    - ```bool SomeFunction()```

 - For non-void functions I usually declare the object being returned as ```out```. Objects that are temporary and have no deeper meaning (i.e. a bond length has a meaning) are often declared as ```temp```.
 

 ```c++
 arma::cube f(){
     arma::cube out(1, 2, 3);
     ...
     arma::vec temp = ...;
     out.col(0) = temp;
     ...
     return out;
   }
```

 - [cpplint.py](https://github.com/google/styleguide/blob/gh-pages/cpplint/cpplint.py) will be run to find cases of incorrect style.

    Run cpplint.py: ```python3 cpplint.py --filter=-legal/copyright --recursive src/``` 
 
 - Static analysis perfomed by ```cppcheck``` (though I have found this does not catch basic errors?)
    ```cppcheck --project=compile_commands.json```
    & ```clang-tidy```
      ```run-clang-tidy-10 -style 'google'```
 - Use clang-format to format the c++ code. The style file (.clang-format) is in the root directory of the project. This should be found when formatting but if not then set -style=file (exactly this, do not replace the word file with an actual filename) or in you VS Code settings: "clang-format.style": "file".
