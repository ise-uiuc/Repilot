# Repilot Patch

```
    public static final CSVFormat EXCEL =
            DEFAULT
            .withIgnoreEmptyLines(false)
            .withAllowMissingColumnNames(true);
```

# Developer Patch

```
    public static final CSVFormat EXCEL = DEFAULT.withIgnoreEmptyLines(false).withAllowMissingColumnNames(true);
```

# Context

```
--- bug/Csv-12/src/main/java/org/apache/commons/csv/CSVFormat.java

+++ fix/Csv-12/src/main/java/org/apache/commons/csv/CSVFormat.java

@@ -216,7 +216,10 @@

      * Note: this is currently like {@link #RFC4180} plus {@link #withAllowMissingColumnNames(boolean) withAllowMissingColumnNames(true)}.
      * </p>
      */
-    public static final CSVFormat EXCEL = DEFAULT.withIgnoreEmptyLines(false);
+    public static final CSVFormat EXCEL =
+            DEFAULT
+            .withIgnoreEmptyLines(false)
+            .withAllowMissingColumnNames(true);
 
     /**
      * Tab-delimited format.
```

# Note

