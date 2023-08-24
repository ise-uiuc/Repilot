# Repilot Patch

```
//        if (property == null && value.value instanceof Map) {
//            return this;
//        }

        // Do we already have an override for this?
```

# Developer Patch

```

```

# Context

```
--- bug/JacksonDatabind-102/src/main/java/com/fasterxml/jackson/databind/ser/std/DateTimeSerializerBase.java

+++ fix/JacksonDatabind-102/src/main/java/com/fasterxml/jackson/databind/ser/std/DateTimeSerializerBase.java

@@ -64,9 +64,11 @@

     {
         // Note! Should not skip if `property` null since that'd skip check
         // for config overrides, in case of root value
-        if (property == null) {
-            return this;
-        }
+//        if (property == null && value.value instanceof Map) {
+//            return this;
+//        }
+
+        // Do we already have an override for this?
         JsonFormat.Value format = findFormatOverrides(serializers, property, handledType());
         if (format == null) {
             return this;
```

# Note

