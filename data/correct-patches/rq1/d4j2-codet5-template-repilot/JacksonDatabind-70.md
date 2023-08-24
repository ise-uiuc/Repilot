# Repilot Patch

```
found = key.equals(_hashArea[i - 1]);
```

# Developer Patch

```
                found = key.equals(_hashArea[i-1]);
```

# Context

```
--- bug/JacksonDatabind-70/src/main/java/com/fasterxml/jackson/databind/deser/impl/BeanPropertyMap.java

+++ fix/JacksonDatabind-70/src/main/java/com/fasterxml/jackson/databind/deser/impl/BeanPropertyMap.java

@@ -437,7 +437,7 @@

             if (!found) {
                 // 09-Jan-2017, tatu: Important: must check name slot and NOT property name,
                 //   as only former is lower-case in case-insensitive case
-                found = key.equals(prop.getName());
+found = key.equals(_hashArea[i - 1]);
                 if (found) {
                     // need to leave a hole here
                     _propsInOrder[_findFromOrdered(prop)] = null;
```

# Note

