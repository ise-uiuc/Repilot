# Repilot Patch

```
return FastMath.pow(2 * FastMath.PI,-0.5 * dim) *
```

# Developer Patch

```
        return FastMath.pow(2 * FastMath.PI, -0.5 * dim) *
```

# Context

```
--- bug/Math-11/src/main/java/org/apache/commons/math3/distribution/MultivariateNormalDistribution.java

+++ fix/Math-11/src/main/java/org/apache/commons/math3/distribution/MultivariateNormalDistribution.java

@@ -180,7 +180,7 @@

             throw new DimensionMismatchException(vals.length, dim);
         }
 
-        return FastMath.pow(2 * FastMath.PI, -dim / 2) *
+return FastMath.pow(2 * FastMath.PI,-0.5 * dim) *
             FastMath.pow(covarianceMatrixDeterminant, -0.5) *
             getExponentTerm(vals);
     }
```

# Note

