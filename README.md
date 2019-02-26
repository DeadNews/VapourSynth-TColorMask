# TColorMask ##

VapourSynth port of tp7's Avisynth plugin. Great solution if you need to build some masks on color values.
Only 8 bit clips are allowed at this time.

### Usage ###
All parameters (except colors) are set to their default values.
```
core.tcm.TColorMask(clip, ['$FFFFFF', '$000000', '$808080'], tolerance=10, bt601=False, gray=False, lutthr=9)
```
* *colors* - array of colors. Required.
* *tolerance* - pixel value will pass if its absolute difference with color is less than tolerance (luma) or half the tolerance (chroma).
* *bt601* - use bt601 matrix for conversion of colors.
* *gray* - set chroma of output clip to 128. Chroma will contain garbage if False.
* *lutthr* - if specified more than lutthr colors, lut will be used instead of direct SIMD computations. 

### License ###
This project is licensed under the [MIT license][mit_license].

[mit_license]: http://opensource.org/licenses/MIT



### Compilation ###

```
mkdir -p build && cd build
cmake ../
cmake --build .
```
