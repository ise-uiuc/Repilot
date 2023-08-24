# Repilot Patch

```
            int j = 4 * (n - 1);
```

# Developer Patch

```
            int j = 4 * (n - 1);
```

# Context

```
--- bug/Math-80/src/main/java/org/apache/commons/math/linear/EigenDecompositionImpl.java

+++ fix/Math-80/src/main/java/org/apache/commons/math/linear/EigenDecompositionImpl.java

@@ -1132,7 +1132,7 @@

     private boolean flipIfWarranted(final int n, final int step) {
         if (1.5 * work[pingPong] < work[4 * (n - 1) + pingPong]) {
             // flip array
-            int j = 4 * n - 1;
+            int j = 4 * (n - 1);
             for (int i = 0; i < j; i += 4) {
                 for (int k = 0; k < 4; k += step) {
                     final double tmp = work[i + k];
```

# Note

