# Repilot Patch

```

        return solve(f, min, max);
```

# Developer Patch

```
        return solve(f, min, max);
```

# Context

```
--- bug/Math-70/src/main/java/org/apache/commons/math/analysis/solvers/BisectionSolver.java

+++ fix/Math-70/src/main/java/org/apache/commons/math/analysis/solvers/BisectionSolver.java

@@ -69,7 +69,8 @@

     /** {@inheritDoc} */
     public double solve(final UnivariateRealFunction f, double min, double max, double initial)
         throws MaxIterationsExceededException, FunctionEvaluationException {
-        return solve(min, max);
+
+        return solve(f, min, max);
     }
 
     /** {@inheritDoc} */
```

# Note

