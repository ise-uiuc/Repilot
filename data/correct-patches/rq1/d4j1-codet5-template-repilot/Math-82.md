# Repilot Patch

```
if (MathUtils.compareTo(entry, 0, epsilon) > 0) {
```

# Developer Patch

```
            if (MathUtils.compareTo(entry, 0, epsilon) > 0) {
```

# Context

```
--- bug/Math-82/src/main/java/org/apache/commons/math/optimization/linear/SimplexSolver.java

+++ fix/Math-82/src/main/java/org/apache/commons/math/optimization/linear/SimplexSolver.java

@@ -79,7 +79,7 @@

         for (int i = tableau.getNumObjectiveFunctions(); i < tableau.getHeight(); i++) {
             final double rhs = tableau.getEntry(i, tableau.getWidth() - 1);
             final double entry = tableau.getEntry(i, col);
-            if (MathUtils.compareTo(entry, 0, epsilon) >= 0) {
+if (MathUtils.compareTo(entry, 0, epsilon) > 0) {
                 final double ratio = rhs / entry;
                 if (ratio < minRatio) {
                     minRatio = ratio;
```

# Note

