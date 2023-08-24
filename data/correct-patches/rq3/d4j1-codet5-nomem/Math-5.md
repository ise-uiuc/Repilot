# Repilot Patch

```

            return INF;
```

# Developer Patch

```
            return INF;
```

# Context

```
--- bug/Math-5/src/main/java/org/apache/commons/math3/complex/Complex.java

+++ fix/Math-5/src/main/java/org/apache/commons/math3/complex/Complex.java

@@ -302,7 +302,8 @@

         }
 
         if (real == 0.0 && imaginary == 0.0) {
-            return NaN;
+
+            return INF;
         }
 
         if (isInfinite) {
```

# Note

