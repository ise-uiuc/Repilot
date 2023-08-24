# Repilot Patch

```
        if (fa * fb > 0.0) {
```

# Developer Patch

```
        if (fa * fb > 0.0 ) {
```

# Context

```
--- bug/Math-85/src/java/org/apache/commons/math/analysis/solvers/UnivariateRealSolverUtils.java

+++ fix/Math-85/src/java/org/apache/commons/math/analysis/solvers/UnivariateRealSolverUtils.java

@@ -195,7 +195,8 @@

         } while ((fa * fb > 0.0) && (numIterations < maximumIterations) && 
                 ((a > lowerBound) || (b < upperBound)));
    
-        if (fa * fb >= 0.0 ) {
+        if (fa * fb > 0.0) {
+            
             throw new ConvergenceException(
                       "number of iterations={0}, maximum iterations={1}, " +
                       "initial={2}, lower bound={3}, upper bound={4}, final a value={5}, " +
```

# Note

