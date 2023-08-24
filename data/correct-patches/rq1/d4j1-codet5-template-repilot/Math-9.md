# Repilot Patch

```
final Line reverted = new Line(this);
        reverted.direction = this.direction.negate();
```

# Developer Patch

```
        final Line reverted = new Line(this);
        reverted.direction = reverted.direction.negate();
```

# Context

```
--- bug/Math-9/src/main/java/org/apache/commons/math3/geometry/euclidean/threed/Line.java

+++ fix/Math-9/src/main/java/org/apache/commons/math3/geometry/euclidean/threed/Line.java

@@ -84,7 +84,8 @@

      * @return a new instance, with reversed direction
      */
     public Line revert() {
-        final Line reverted = new Line(zero, zero.subtract(direction));
+final Line reverted = new Line(this);
+        reverted.direction = this.direction.negate();
         return reverted;
     }
```

# Note

