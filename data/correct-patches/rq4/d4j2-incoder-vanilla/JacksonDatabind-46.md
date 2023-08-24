# Repilot Patch

```
        sb.append(">;");
```

# Developer Patch

```
        sb.append(">;");
```

# Context

```
--- bug/JacksonDatabind-46/src/main/java/com/fasterxml/jackson/databind/type/ReferenceType.java

+++ fix/JacksonDatabind-46/src/main/java/com/fasterxml/jackson/databind/type/ReferenceType.java

@@ -153,7 +153,7 @@

         _classSignature(_class, sb, false);
         sb.append('<');
         sb = _referencedType.getGenericSignature(sb);
-        sb.append(';');
+        sb.append(">;");
         return sb;
     }
```

# Note

