# Repilot Patch

```

        return this.headerMap == null ? null : new LinkedHashMap<String, Integer>(this.headerMap);
```

# Developer Patch

```
        return this.headerMap == null ? null : new LinkedHashMap<String, Integer>(this.headerMap);
```

# Context

```
--- bug/Csv-4/src/main/java/org/apache/commons/csv/CSVParser.java

+++ fix/Csv-4/src/main/java/org/apache/commons/csv/CSVParser.java

@@ -285,7 +285,8 @@

      * @return a copy of the header map that iterates in column order.
      */
     public Map<String, Integer> getHeaderMap() {
-        return new LinkedHashMap<String, Integer>(this.headerMap);
+
+        return this.headerMap == null ? null : new LinkedHashMap<String, Integer>(this.headerMap);
     }
 
     /**
```

# Note

