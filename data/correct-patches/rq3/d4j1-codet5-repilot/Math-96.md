# Repilot Patch

```

                    ret = this.real == rhs.real && this.imaginary == rhs.imaginary;
```

# Developer Patch

```
                    ret = (real == rhs.real) && (imaginary == rhs.imaginary);
```

# Context

```
--- bug/Math-96/src/java/org/apache/commons/math/complex/Complex.java

+++ fix/Math-96/src/java/org/apache/commons/math/complex/Complex.java

@@ -255,7 +255,8 @@

                 if (rhs.isNaN()) {
                     ret = this.isNaN();
                 } else {
-                    ret = (Double.doubleToRawLongBits(real) == Double.doubleToRawLongBits(rhs.getReal())) && (Double.doubleToRawLongBits(imaginary) == Double.doubleToRawLongBits(rhs.getImaginary())); 
+
+                    ret = this.real == rhs.real && this.imaginary == rhs.imaginary;
                 }
             } catch (ClassCastException ex) {
                 // ignore exception
```

# Note

