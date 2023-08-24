# Repilot Patch

```

```

# Developer Patch

```

```

# Context

```
--- bug/Math-50/src/main/java/org/apache/commons/math/analysis/solvers/BaseSecantSolver.java

+++ fix/Math-50/src/main/java/org/apache/commons/math/analysis/solvers/BaseSecantSolver.java

@@ -184,10 +184,7 @@

                     break;
                 case REGULA_FALSI:
                     // Nothing.
-                    if (x == x1) {
-                        x0 = 0.5 * (x0 + x1 - FastMath.max(rtol * FastMath.abs(x1), atol));
-                        f0 = computeObjectiveValue(x0);
-                    }
+
                     break;
                 default:
                     // Should never happen.
```

# Note

