# Repilot Patch

```

        if (raw.isAssignableFrom(String.class)) {
```

# Developer Patch

```
        if (raw == String.class || raw == Object.class || raw == CharSequence.class) {
```

# Context

```
--- bug/JacksonDatabind-71/src/main/java/com/fasterxml/jackson/databind/deser/std/StdKeyDeserializer.java

+++ fix/JacksonDatabind-71/src/main/java/com/fasterxml/jackson/databind/deser/std/StdKeyDeserializer.java

@@ -72,7 +72,8 @@

         int kind;
 
         // first common types:
-        if (raw == String.class || raw == Object.class) {
+
+        if (raw.isAssignableFrom(String.class)) {
             return StringKD.forType(raw);
         } else if (raw == UUID.class) {
             kind = TYPE_UUID;
```

# Note

