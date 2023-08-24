# Repilot Patch

```
int g = (int) (((v - this.lowerBound) / this.upperBound
```

# Developer Patch

```
int g = (int) ((v - this.lowerBound) / (this.upperBound
```

# Context

```
--- bug/Chart-24/source/org/jfree/chart/renderer/GrayPaintScale.java

+++ fix/Chart-24/source/org/jfree/chart/renderer/GrayPaintScale.java

@@ -123,7 +123,8 @@

     public Paint getPaint(double value) {
         double v = Math.max(value, this.lowerBound);
         v = Math.min(v, this.upperBound);
-        int g = (int) ((value - this.lowerBound) / (this.upperBound 
+
+        int g = (int) (((v - this.lowerBound) / this.upperBound
                 - this.lowerBound) * 255.0);
         return new Color(g, g, g);
     }
```

# Note

