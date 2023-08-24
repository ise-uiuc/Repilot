# Repilot Patch

```
        if (len == 1) {
            return a[0] * b[0];
        }

        // Use Stern-Brocot Algorithm
```

# Developer Patch

```
        if (len == 1) {
            return a[0] * b[0];
        }
```

# Context

```
--- bug/Math-3/src/main/java/org/apache/commons/math3/util/MathArrays.java

+++ fix/Math-3/src/main/java/org/apache/commons/math3/util/MathArrays.java

@@ -818,7 +818,11 @@

             throw new DimensionMismatchException(len, b.length);
         }
 
-            // Revert to scalar multiplication.
+        if (len == 1) {
+            return a[0] * b[0];
+        }
+
+        // Use Stern-Brocot Algorithm
 
         final double[] prodHigh = new double[len];
         double prodLowSum = 0;
```

# Note

