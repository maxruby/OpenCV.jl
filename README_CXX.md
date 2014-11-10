##CXX.jl tutorial


Wrapping C++ functions in Julia ([Cxx.jl](https://github.com/Keno/Cxx.jl)) 

### Installation

```sh
# You will need to install Julia v0.4-dev
# Cxx.jl requires `staged functions` available only in v0.4 
# switch to the julia-v0.4-dev directory:

$ make -C deps distclean-openblas distclean-arpack distclean-suitesparse && make cleanall 
$ make –j4
``` 

In the Julia terminal, type
```python
Pkg.clone("https://github.com/Keno/Cxx.jl.git")
Pkg.build("Cxx")   
```

### Introduction 

To embedd C++ functions in Julia, there are currently two main approaches:

```python  
# Using `@cxx` macro:   
cxx""" void cppfunction(args){ . . .} """ => @cxx cppfunction(args)

# Using `icxx` call:
julia_function (args) icxx""" *code here*  """ end
```    

### **Using Cxx:** 

#### Example 1: Simple math function 

```python
# include headers
julia> using Cxx
julia> cxx""" #include<iostream> """  

# Declare the function
julia> cxx"""  
         void mycppfunction1() {   
            int z = 0;
            int y = 5;
            int x = 10;
            z = x*y + 2;
            std::cout << "The number is " << z << std::endl;
         }
      """
# Convert C++ to Julia function
julia> julia_function() = @cxx mycppfunction1()
julia_function (generic function with 1 method)
   
# Run the function
julia> julia_function()
julia_to_llvm(Void) = CppPtr{symbol("llvm::Type"),()}(Ptr{Void}@0x00007fa87b0002c8)
argt = Any[]
The number is 52
```

#### Example 2: Numeric input from Julia to C++

```python
julia> jnum = 10
10
    
julia> cxx"""
           void printme(int x) {
              std::cout << x << std::endl;
           }
       """
       
julia> @cxx printme2(jnum)
julia_to_llvm(Void) = CppPtr{symbol("llvm::Type"),()}(Ptr{Void} 
@0x00007fa87b0002c8)
argt = [CppPtr{symbol("llvm::Type"),()}(Ptr{Void} @0x00007fa87b000418)] 
10 
```

#### Example 3: String input from Julia to C++  
 ```python
julia> cxx"""
          void printme(const char *name) {
             // const char* => std::string
             std::string sname = name;
             // print it out
             std::cout << sname << std::endl;
          }
      """

julia> @cxx printme(pointer("Maximiliano"))
    julia_to_llvm(Void) = CppPtr{symbol("llvm::Type"),()}(Ptr{Void}
    @0x00007fa87b0002c8)
    argt = [CppPtr{symbol("llvm::Type"),()}(Ptr{Void}@0x00007fa87b0a6e00)]
    Maximiliano 
```

#### Example 4: Passing an expression from Julia to C++

```python
julia> cxx"""
          void testJuliaPrint() {
              $:(println("\nTo end this test, press any key")::Nothing);
          }
       """

julia> @cxx testJuliaPrint()
       julia_to_llvm(Void) = CppPtr{symbol("llvm::Type"),()}(Ptr{Void}@0x00007fa87b0002c8)
       argt = Any[]

       To end this test, press any key
```

#### Example 5: Embedding C++ code in Julia functions

```python
function playing()
    for i = 1:5
        icxx"""
            int tellme;
            std::cout<< "Please enter a number: " << std::endl;
            std::cin >> tellme;
            std::cout<< "\nYour number is "<< tellme << "\n" <<std::endl;
        """
    end
end
playing();
```

