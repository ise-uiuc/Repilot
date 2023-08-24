# Repilot Patch

```
if (dataset == null) {
```

# Developer Patch

```
if (dataset == null) {
```

# Context

```
--- bug/Chart-1/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java

+++ fix/Chart-1/source/org/jfree/chart/renderer/category/AbstractCategoryItemRenderer.java

@@ -1794,7 +1794,8 @@

         }
         int index = this.plot.getIndexOf(this);
         CategoryDataset dataset = this.plot.getDataset(index);
-        if (dataset != null) {
+
+        if (dataset == null) {
             return result;
         }
         int seriesCount = dataset.getRowCount();
```

# Note

