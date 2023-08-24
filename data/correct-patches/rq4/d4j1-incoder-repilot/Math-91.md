# Repilot Patch

```
        long dOn = this.denominator * object.numerator;
        long nOd = this.numerator * object.denominator;
```

# Developer Patch

```
        long nOd = ((long) numerator) * object.denominator;
        long dOn = ((long) denominator) * object.numerator;
```

# Context

```
--- bug/Math-91/src/java/org/apache/commons/math/fraction/Fraction.java

+++ fix/Math-91/src/java/org/apache/commons/math/fraction/Fraction.java

@@ -256,8 +256,8 @@

      *         than <tt>object</tt>, 0 if they are equal.
      */
     public int compareTo(Fraction object) {
-        double nOd = doubleValue();
-        double dOn = object.doubleValue();
+        long dOn = this.denominator * object.numerator;
+        long nOd = this.numerator * object.denominator;
         return (nOd < dOn) ? -1 : ((nOd > dOn) ? +1 : 0);
     }
```

# Note

