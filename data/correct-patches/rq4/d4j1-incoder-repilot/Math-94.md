# Repilot Patch

```
        // assert(Math.abs(u-v)<=1);  // if |...| <= 1, gcd == 1
        if (u == 0 || v == 0) {
```

# Developer Patch

```
        if ((u == 0) || (v == 0)) {
```

# Context

```
--- bug/Math-94/src/java/org/apache/commons/math/util/MathUtils.java

+++ fix/Math-94/src/java/org/apache/commons/math/util/MathUtils.java

@@ -409,7 +409,8 @@

      * @since 1.1
      */
     public static int gcd(int u, int v) {
-        if (u * v == 0) {
+        // assert(Math.abs(u-v)<=1);  // if |...| <= 1, gcd == 1
+        if (u == 0 || v == 0) {
             return (Math.abs(u) + Math.abs(v));
         }
         // keep u and v negative, as negative integers range down to
```

# Note

