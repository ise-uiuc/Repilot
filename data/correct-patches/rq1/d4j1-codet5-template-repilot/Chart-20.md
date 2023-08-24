# Repilot Patch

```
super(paint, stroke, outlinePaint, outlineStroke, alpha);
```

# Developer Patch

```
        super(paint, stroke, outlinePaint, outlineStroke, alpha);
```

# Context

```
--- bug/Chart-20/source/org/jfree/chart/plot/ValueMarker.java

+++ fix/Chart-20/source/org/jfree/chart/plot/ValueMarker.java

@@ -92,7 +92,7 @@

      */
     public ValueMarker(double value, Paint paint, Stroke stroke, 
                        Paint outlinePaint, Stroke outlineStroke, float alpha) {
-        super(paint, stroke, paint, stroke, alpha);
+super(paint, stroke, outlinePaint, outlineStroke, alpha);
         this.value = value;
     }
```

# Note

