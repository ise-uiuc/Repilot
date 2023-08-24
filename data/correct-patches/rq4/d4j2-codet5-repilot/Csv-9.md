# Repilot Patch

```

        if (mapping == null) {
            return map;
        }
```

# Developer Patch

```
        if (mapping == null) {
            return map;
        }
```

# Context

```
--- bug/Csv-9/src/main/java/org/apache/commons/csv/CSVRecord.java

+++ fix/Csv-9/src/main/java/org/apache/commons/csv/CSVRecord.java

@@ -177,6 +177,10 @@

      * @return the given map.
      */
     <M extends Map<String, String>> M putIn(final M map) {
+
+        if (mapping == null) {
+            return map;
+        }
         for (final Entry<String, Integer> entry : mapping.entrySet()) {
             final int col = entry.getValue().intValue();
             if (col < values.length) {
```

# Note

