# Repilot Patch

```

        // Check for formats:
```

# Developer Patch

```

```

# Context

```
--- bug/JacksonDatabind-102/src/main/java/com/fasterxml/jackson/databind/ser/std/DateTimeSerializerBase.java

+++ fix/JacksonDatabind-102/src/main/java/com/fasterxml/jackson/databind/ser/std/DateTimeSerializerBase.java

@@ -64,9 +64,8 @@

     {
         // Note! Should not skip if `property` null since that'd skip check
         // for config overrides, in case of root value
-        if (property == null) {
-            return this;
-        }
+
+        // Check for formats:
         JsonFormat.Value format = findFormatOverrides(serializers, property, handledType());
         if (format == null) {
             return this;
```

# Note

