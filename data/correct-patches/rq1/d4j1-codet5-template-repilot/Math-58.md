# Repilot Patch

```
return fit(guess);
```

# Developer Patch

```
        return fit(guess);
```

# Context

```
--- bug/Math-58/src/main/java/org/apache/commons/math/optimization/fitting/GaussianFitter.java

+++ fix/Math-58/src/main/java/org/apache/commons/math/optimization/fitting/GaussianFitter.java

@@ -118,7 +118,7 @@

      */
     public double[] fit() {
         final double[] guess = (new ParameterGuesser(getObservations())).guess();
-        return fit(new Gaussian.Parametric(), guess);
+return fit(guess);
     }
 
     /**
```

# Note

