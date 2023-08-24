# Repilot Patch

```
return this.doubleValue() * 100;
```

# Developer Patch

```
return 100 * doubleValue();
```

# Context

```
--- bug/Math-27/src/main/java/org/apache/commons/math3/fraction/Fraction.java

+++ fix/Math-27/src/main/java/org/apache/commons/math3/fraction/Fraction.java

@@ -594,7 +594,8 @@

      * @return the fraction percentage as a <tt>double</tt>.
      */
     public double percentageValue() {
-        return multiply(100).doubleValue();
+
+        return this.doubleValue() * 100;
     }
 
     /**
```

# Note

